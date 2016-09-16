import WCS
import Base.convert

# CatalogEntry size is 201 (pos:2, star_fluxes:5, gal_fluxes:5, objid:19)
# RawPSF size is 84060 (rrows: 2601x4, cmat: 4x5x5)
# PsfComponent size is 150 (xiBar: 2, tauBar:2x2, tauBarInv:2x2)
# ImageTile size is 3296 (pixels: 20x20, epsilon_mat: 20x20, iota_vec: 20)
# TiledImage size is 69394456 (tiles: 105x200, wcs_header: 10000, psf: 2)
# InferResult size is 299 (objid: 19, vs: 32)

immutable InferResult
    thing_id::Int
    objid::String
    ra::Float64
    dec::Float64
    vs::Vector{Float64}
end


function fetch_catalog(rcf, stagedir)
    # note: this call to read_photoobj_files considers only primary detections.
    catalog = SDSSIO.read_photoobj_files([rcf,], stagedir)

    # we're ignoring really faint sources entirely...not even using them to
    # render the background
    isnt_faint(entry) = maximum(entry.star_fluxes) >= MIN_FLUX
    filter(isnt_faint, catalog)
end


@inline function in_box(entry::CatalogEntry, box::BoundingBox)
    box.ramin < entry.pos[1] < box.ramax &&
        box.decmin < entry.pos[2] < box.decmax
end


function load_images(box, rcfs, stagedir)
    num_fields = length(rcfs)
    nputs(nodeid, "$num_fields RCFs")

    # each cell of `images` contains B=5 tiled images
    images = Garray(NTuple{5,TiledImage}, 347500000, num_fields)

    # stores first index of each field's sources in the catalog array
    catalog_offset = Garray(Int64, 10, num_fields)

    # stores first index of each field's tasks in the tasks array
    task_offset = Garray(Int64, 10, num_fields)

    # get local distribution of the global array; this should be identical
    # for all the arrays (since they're all the same size)
    lo, hi = distribution(images, nodeid)
    nlocal = hi[1]-lo[1]+1
    nputs(nodeid, "$nlocal local images ($(lo[1])-$(hi[1]))")

    # get access to the local parts of the global arrays
    limages = access(images, lo, hi)
    lcatalog_offset = access(catalog_offset, lo, hi)
    ltask_offset = access(task_offset, lo, hi)

    for i in 1:nlocal
        n = lo[1] + i - 1

        rcf = rcfs[n]
        nputs(nodeid, "loading images for $(rcf.run), $(rcf.camcol), $(rcf.field)")
        raw_images = SDSSIO.load_field_images(rcf, stagedir)
        @assert(length(raw_images) == 5)
        timgs = [TiledImage(img) for img in raw_images]
        limages[i] = tuple(timgs...)
   
        # second, load the `catalog_offset` and `task_count` arrays with
        # a number of sources for each field.
        # (We'll accumulate the entries later.)
        local_catalog = fetch_catalog(rcf, stagedir)

        # we'll use sources outside of the box to render the background,
        # but we won't optimize them
        local_tasks = filter(s->in_box(s, box), local_catalog)

        lcatalog_offset[i] = length(local_catalog)
        ltask_offset[i] = length(local_tasks)
    end
    flush(images)
    flush(catalog_offset)
    flush(task_offset)
    sync()

    # folds right, converting each field's count to offsets in # `catalog`
    # and `tasks`. this is a prefix sum, which can be done in parallel
    # (TODO: add this to Garbo) but is being done sequentially here
    for nid = 1:nnodes
        # one node at a time
        if nid == nodeid
            # the first node has no predecessor
            if nid == 1
                catalog_size = 0
                num_tasks = 0
            # all other nodes have to get the sum from their predecessor
            else
                prev = get(catalog_offset, lo-1, lo-1)
                catalog_size = prev[1]
                prev = get(task_offset, lo-1, lo-1)
                num_tasks = prev[1]
            end
            # do the local summing
            for i in 1:nlocal
                catalog_size += lcatalog_offset[i]
                lcatalog_offset[i] = catalog_size
                num_tasks += ltask_offset[i]
                ltask_offset[i] = num_tasks
            end
        end
        # sync to ensure orderly progression
        flush(catalog_offset)
        flush(task_offset)
        sync()
    end

    images, catalog_offset, task_offset
end


function load_catalog(box, rcfs, catalog_offset, task_offset, stagedir)
    num_fields = length(rcfs)
    coe = get(catalog_offset, [num_fields], [num_fields])
    catalog_size = coe[1]
    nputs(nodeid, "catalog size is $catalog_size")
    toe = get(task_offset, [num_fields], [num_fields])
    num_tasks = toe[1]
    nputs(nodeid, "$num_tasks tasks")

    catalog = Garray(Tuple{CatalogEntry, RunCamcolField}, 300, catalog_size)

    # entries in `tasks` are indexes into `catalog`
    tasks = Garray(Int64, 10, num_tasks)

    # get local distribution of the arrays
    colo, cohi = distribution(catalog_offset, nodeid)
    nlocal = cohi[1]-colo[1]+1
    clo, chi = distribution(catalog, nodeid)
    lcatalog_size = chi[1]-clo[1]+1
    nputs(nodeid, "$lcatalog_size local catalog entries")
    tlo, thi = distribution(tasks, nodeid)
    ltasks_size = thi[1]-tlo[1]+1

    # get access to the local parts of the global arrays
    lcatalog = access(catalog, clo, chi)
    ltasks = access(tasks, tlo, thi)

    # iterate over the local catalog offsets to build the catalog and tasks list
    for i in 1:nlocal
        n = colo[1] + i - 1
        rcf = rcfs[n]

        local_catalog = fetch_catalog(rcf, stagedir)

        local_t = 0
        for local_s in 1:length(local_catalog)
            entry = local_catalog[local_s]
            # `s` is the global index of this source (in the catalog)
            offset = n == 1 ? 0 : catalog_offset[n-1]
            s = offset + local_s
            # each field's primary detection is stored contiguously
            #catalog[s] = (entry, rcf)
            put!(catalog, [s], [s], [(entry, rcf)])

            if in_box(entry, box)
                local_t += 1
                offset = n == 1 ? 0 : task_offset[n - 1]
                t = offset + local_t
                #tasks[t] = s
                put!(tasks, [t], [t], [s])
            end
        end
    end

    catalog, tasks
end


function invert_rcf_array(rcfs)
    max_run = 1
    max_camcol = 1
    max_field = 1
    for rcf in rcfs
        max_run = max(max_run, rcf.run)
        max_camcol = max(max_camcol, rcf.camcol)
        max_field = max(max_field, rcf.field)
    end

    rcf_to_index = zeros(Int64, max_run, max_camcol, max_field)

    # this should be really fast
    for n in 1:length(rcfs)
        rcf = rcfs[n]
        rcf_to_index[rcf.run, rcf.camcol, rcf.field] = n
    end

    rcf_to_index
end


function optimize_source(s, images, catalog, catalog_offset, rcf_to_index,
                         stagedir, results)
    local_images = Vector{TiledImage}()
    local_catalog = CatalogEntry[];

    ep = get(catalog, [s], [s])
    entry, primary_rcf = ep[1]
    t_box = BoundingBox(entry.pos[1] - 1e-8, entry.pos[1] + 1e-8,
                        entry.pos[2] - 1e-8, entry.pos[2] + 1e-8)
    surrounding_rcfs = get_overlapping_fields(t_box, stagedir)

    for rcf in surrounding_rcfs
        n = rcf_to_index[rcf.run, rcf.camcol, rcf.field]
        #@assert n > 0
        nputs(nodeid, "getting image $n for $(rcf.run), $(rcf.camcol), $(rcf.field)")
        imgs = get(images, [n], [n])
        push!(local_images, imgs[1]...)
        if n == 1
            s_a = 1
            st = get(catalog_offset, [n], [n])
            s_b = st[1]
        else
            st = get(catalog_offset, [n-1], [n])
            s_a = st[1]
            s_b = st[2]
        end
        nputs(nodeid, "s_a=$s_a, s_b=$s_b")
        neighbors = get(catalog, [s_a], [s_b])
        for neighbor in neighbors
            push!(local_catalog, neighbor[1])
        end
    end

    #flat_images = [img for img5 in local_images for img in img5]

    i = findfirst(local_catalog, entry)
    nputs(nodeid, "i=$i")
    neighbor_indexes = Infer.find_neighbors([i,], local_catalog, local_images)[1]
    neighbors = local_catalog[neighbor_indexes]

    vs_opt = Infer.infer_source(local_images, neighbors, entry)

    put!(results, [s], [s], [InferResult(entry.thing_id, entry.objid,
                                    entry.pos[1], entry.pos[2], vs_opt)])
end


function optimize_sources(images, catalog, tasks, catalog_offset, task_offset,
            rcf_to_index, stagedir, results, timing)
    num_work_items = length(tasks)
    if nodeid == 1
        nputs(nodeid, "processing $num_work_items work items on $nnodes nodes")
    end

    # create Dtree and get the initial allocation
    dt, isparent = Dtree(num_work_items, 0.4)
    ni, (ci, li) = initwork(dt)
    nputs(nodeid, "got $ni tasks ($ci-$li)")
    rundt = runtree(dt)
    wi = get(tasks, [ci], [li])
    ci = 1
    li = ni
    wilock = SpinLock()

    function process_tasks()
        tid = threadid()
        if rundt && tid == 1
            ntputs(nodeid, tid, "running tree")
            while runtree(dt)
                cpu_pause()
            end
        else
            while true
                lock(wilock)
                if li == 0
                    ntputs(nodeid, tid, "out of work")
                    unlock(wilock)
                    break
                end
                if ci == li
                    ntputs(nodeid, tid, "consumed last work item ($item); requesting more")
                    ni, (ci, li) = getwork(dt)
                    wi = get(tasks, [ci], [li])
                    ci = 1
                    li = ni
                    ntputs(nodeid, tid, "got $ni work items")
                    unlock(wilock)
                    continue
                end
                item = wi[ci]
                ci = ci + 1
                unlock(wilock)
                ntputs(nodeid, tid, "processing item $item")

                optimize_source(item, images, catalog, catalog_offset, rcf_to_index,
                                stagedir, results)
            end
        end
    end

    tic()
    #ccall(:jl_threading_run, Void, (Any,), Core.svec(process_tasks))
    process_tasks()
    timing.opt_srcs = toq()

    if nodeid == 1
        nputs(nodeid, "complete")
    end
    tic()
    finalize(dt)
    timing.wait_done = toq()
end


"""
Fit the Celeste model to sources in a given ra, dec range,
based on data from specified fields
- box: a bounding box specifying a region of sky
"""
function divide_sources_and_infer(
                box::BoundingBox,
                stagedir::String;
                timing=InferTiming(),
                outdir=".")
    # read the run-camcol-field triplets for this box
    tic()
    rcfs = get_overlapping_fields(box, stagedir)
    timing.query_fids = toq()

    # loads 25TB from disk for SDSS
    tic()
    images, catalog_offset, task_offset = load_images(box, rcfs, stagedir)
    timing.read_img = toq()

    # loads 4TB from disk for SDSS
    tic()
    catalog, tasks = load_catalog(box, rcfs, catalog_offset, task_offset, stagedir)
    timing.read_photoobj = toq()

    # create map from run, camcol, field to index into RCF array
    rcf_to_index = invert_rcf_array(rcfs)

    # inference results are written here
    results = Garray(InferResult, 350, length(catalog))

    # optimization -- little disk access, cpu intensive
    timing.num_srcs = length(tasks)
    optimize_sources(images, catalog, tasks, catalog_offset, task_offset,
            rcf_to_index, stagedir, results, timing)
end


