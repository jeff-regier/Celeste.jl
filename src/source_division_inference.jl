import WCS
import Base.convert, Base.serialize, Base.deserialize

# CatalogEntry size is 201 (pos:2, star_fluxes:5, gal_fluxes:5, objid:19)
# RawPSF size is 84060 (rrows: 2601x4, cmat: 4x5x5)
# PsfComponent size is 150 (xiBar: 2, tauBar:2x2, tauBarInv:2x2)
# FlatImage size is 24903682 (pixels: 2048x1489, epsilon_mat: 1489, iota_vec: 2048)
# ImageTile size is 3296 (pixels: 20x20, epsilon_mat: 20x20, iota_vec: 20)
# InferResult size is 344 (objid: 19, vs: 32)

immutable FlatImage
    H::Int
    W::Int
    pixels::Matrix{Float32}
    b::Int
    wcs_header::String
    psf::Vector{PsfComponent}
    run_num::Int
    camcol_num::Int
    field_num::Int
    epsilon_mat::Array{Float32, 2}
    iota_vec::Array{Float32, 1}
    raw_psf_comp::RawPSF
end

function convert(::Type{FlatImage}, img::Image)
    wcs_header = WCS.to_header(img.wcs)
    FlatImage(img.H, img.W, img.pixels, img.b,
              wcs_header, img.psf, img.run_num,
              img.camcol_num, img.field_num,
              img.epsilon_mat, img.iota_vec,
              img.raw_psf_comp)
end

function convert(::Type{Image}, img::FlatImage)
    wcs_array = WCS.from_header(img.wcs_header)
    @assert(length(wcs_array) == 1)
    wcs = wcs_array[1]
    Image(img.H, img.W, img.pixels, img.b,
              wcs, img.psf, img.run_num,
              img.camcol_num, img.field_num,
              img.epsilon_mat, img.iota_vec,
              img.raw_psf_comp)
end

function ser_array(s::Base.AbstractSerializer, a::Array, flen::Int)
    serialize(s, size(a))
    for p in a
        write(s.io, p)
    end
    for i in length(a)+1:flen
        write(s.io, zero(eltype(a)))
    end
end

function deser_array(s::Base.AbstractSerializer, T::DataType, flen::Int)
    dims = deserialize(s)
    a = unsafe_wrap(Array, convert(Ptr{T}, pointer(s.io.data, position(s.io)+1)), dims)
    seek(s.io, position(s.io)+flen*sizeof(T))
    return a
end

function serialize(s::Base.AbstractSerializer, psf::PsfComponent)
    Base.serialize_type(s, typeof(psf))
    write(s.io, psf.alphaBar)
    ser_array(s, psf.xiBar, 2)
    ser_array(s, psf.tauBar, 4)
    ser_array(s, psf.tauBarInv, 4)
    write(s.io, psf.tauBarLd)
end

function deserialize(s::Base.AbstractSerializer, t::Type{PsfComponent})
    alphaBar = read(s.io, Float64)::Float64
    xiBar = deser_array(s, Float64, 2)
    tauBar = deser_array(s, Float64, 4)
    tauBarInv = deser_array(s, Float64, 4)
    tauBarLd = read(s.io, Float64)::Float64
    PsfComponent(alphaBar, xiBar, tauBar, tauBarInv, tauBarLd)
end

function serialize(s::Base.AbstractSerializer, rp::RawPSF)
    Base.serialize_type(s, typeof(rp))
    ser_array(s, rp.rrows, 10500)
    write(s.io, rp.rnrow)
    write(s.io, rp.rncol)
    ser_array(s, rp.cmat, 100)
end

function deserialize(s::Base.AbstractSerializer, t::Type{RawPSF})
    rrows = deser_array(s, Float64, 10500)
    rnrow = read(s.io, Int)::Int
    rncol = read(s.io, Int)::Int
    cmat = deser_array(s, Float64, 100)
    RawPSF(rrows, rnrow, rncol, cmat)
end

function serialize(s::Base.AbstractSerializer, img::FlatImage)
    Base.serialize_type(s, typeof(img))
    write(s.io, img.H)
    write(s.io, img.W)
    ser_array(s, img.pixels, 3100000)
    write(s.io, img.b)
    whlen = length(img.wcs_header.data)
    write(s.io, whlen)
    for i in 1:whlen
        write(s.io, img.wcs_header.data[i])
    end
    for i in whlen+1:10000
        write(s.io, zero(UInt8))
    end
    write(s.io, length(img.psf))
    for p in img.psf
        serialize(s, p)
    end
    write(s.io, img.run_num)
    write(s.io, img.camcol_num)
    write(s.io, img.field_num)
    ser_array(s, img.epsilon_mat, 3100000)
    ser_array(s, img.iota_vec, 2100)
    serialize(s, img.raw_psf_comp)
end

function deserialize(s::Base.AbstractSerializer, t::Type{FlatImage})
    H = read(s.io, Int)::Int
    W = read(s.io, Int)::Int
    pixels = deser_array(s, Float32, 3100000)
    b = read(s.io, Int)::Int
    whlen = read(s.io, Int)::Int
    wcs_header = unsafe_wrap(String, pointer(s.io.data, position(s.io)+1), whlen)
    seek(s.io, position(s.io)+10000)
    psf_len = read(s.io, Int)::Int
    psf = Vector{PsfComponent}(psf_len)
    for i in 1:psf_len
        psf[i] = deserialize(s)
    end
    run_num = read(s.io, Int)::Int
    camcol_num = read(s.io, Int)::Int
    field_num = read(s.io, Int)::Int
    epsilon_mat = deser_array(s, Float32, 3100000)
    iota_vec = deser_array(s, Float32, 2100)
    raw_psf_comp = deserialize(s)
    FlatImage(H, W, pixels, b, wcs_header, psf, run_num, camcol_num,
              field_num, epsilon_mat, iota_vec, raw_psf_comp)
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
    if nodeid == 1
        nputs(nodeid, "$num_fields RCFs")
    end

    # each cell of `images` contains B=5 tiled images
    images = Garray(NTuple{5,FlatImage}, 124780544, num_fields)

    # stores first index of each field's sources in the catalog array
    catalog_offset = Garray(Int64, 10, num_fields)

    # stores first index of each field's tasks in the tasks array
    task_offset = Garray(Int64, 10, num_fields)

    # get local distribution of the global array; this should be identical
    # for all the arrays (since they're all the same size)
    lo, hi = distribution(images, nodeid)
    nlocal = hi[1]-lo[1]+1

    # get access to the local parts of the global arrays
    limages = access(images, lo, hi)
    lcatalog_offset = access(catalog_offset, lo, hi)
    ltask_offset = access(task_offset, lo, hi)

    for i in 1:nlocal
        n = lo[1] + i - 1
        rcf = rcfs[n]

        nputs(nodeid, "loading images for RCF $n ($(rcf.run), $(rcf.camcol), $(rcf.field))")
        raw_images = SDSSIO.load_field_images(rcf, stagedir)
        @assert(length(raw_images) == 5)
        fimgs = [FlatImage(img) for img in raw_images]
        limages[i] = tuple(fimgs...)
   
        # second, load the `catalog_offset` and `task_count` arrays with
        # a number of sources for each field.
        # (We'll accumulate the entries later.)
        local_catalog = fetch_catalog(rcf, stagedir)

        # we'll use sources outside of the box to render the background,
        # but we won't optimize them
        local_tasks = filter(s->in_box(s, box), local_catalog)

        nputs(nodeid, "$(length(local_catalog)) sources ($(length(local_tasks)) tasks) in this RCF")

        lcatalog_offset[i] = length(local_catalog)
        ltask_offset[i] = length(local_tasks)
    end
    flush(images)
    flush(catalog_offset)
    flush(task_offset)
    sync()
    lcatalog_offset = access(catalog_offset, lo, hi)
    ltask_offset = access(task_offset, lo, hi)

    # folds right, converting each field's count to offsets in # `catalog`
    # and `tasks`. this is a prefix sum, which can be done in parallel
    # (TODO: add this to Garbo) but is being done sequentially here
    for nid = 1:nnodes
        # one node at a time
        if nid == nodeid && nlocal > 0
            catalog_size = 0
            num_tasks = 0
            if nid > 1
                coa, coa_handle = get(catalog_offset, lo-1, lo-1)
                #nputs(nodeid, "got $(coa[1]) from $(lo[1]-1)")
                catalog_size = catalog_size + coa[1]
                toa, toa_handle = get(task_offset, lo-1, lo-1)
                num_tasks = num_tasks + toa[1]
            end

            # do the local summing
            for i in 1:nlocal
                catalog_size = catalog_size + lcatalog_offset[i]
                lcatalog_offset[i] = catalog_size
                num_tasks = num_tasks + ltask_offset[i]
                ltask_offset[i] = num_tasks
            end
            flush(catalog_offset)
            flush(task_offset)
        end
        # sync to ensure orderly progression
        sync()
    end

    images, catalog_offset, task_offset
end


function load_catalog(box, rcfs, catalog_offset, task_offset, stagedir)
    num_fields = length(rcfs)
    coa, coa_handle = get(catalog_offset, [num_fields], [num_fields])
    catalog_size = coa[1]
    toa, toa_handle = get(task_offset, [num_fields], [num_fields])
    num_tasks = toa[1]

    if nodeid == 1
        nputs(nodeid, "catalog size is $catalog_size, $num_tasks tasks")
    end

    catalog = Garray(Tuple{CatalogEntry, RunCamcolField}, 300, catalog_size)

    # entries in `tasks` are indexes into `catalog`
    tasks = Garray(Int64, 10, num_tasks)

    # we'll iterate over the local parts of the catalog_offset and
    # task_offset arrays to build the catalog and task arrays. our starting
    # indices into the catalog and task arrays are computed from the
    # previous entries in the respective offset arrays.
    colo, cohi = distribution(catalog_offset, nodeid)
    cat_idx = 1
    task_idx = 1
    if colo[1] > 1
        coa, coa_handle = get(catalog_offset, [colo[1]-1], [colo[1]-1])
        cat_idx = cat_idx + coa[1]
        toa, toa_handle = get(task_offset, [colo[1]-1], [colo[1]-1])
        task_idx = task_idx + toa[1]
    end
    for n in colo[1]:cohi[1]
        rcf = rcfs[n]
        rcf_catalog = fetch_catalog(rcf, stagedir)
        for ci = 1:length(rcf_catalog)
            entry = rcf_catalog[ci]
            put!(catalog, [cat_idx], [cat_idx], [(entry, rcf)])
            if in_box(entry, box)
                put!(tasks, [task_idx], [task_idx], [cat_idx])
                task_idx = task_idx + 1
            end
            cat_idx = cat_idx + 1
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


function optimize_source(s::Int64, images::Garray, catalog::Garray,
                         catalog_offset::Garray, rcf_to_index::Array{Int64,3},
                         rcf_cache::Dict, rcf_cache_lock::SpinLock,
                         stagedir::String, results::Garray)
    lock(rcf_cache_lock)
    ep, ep_handle = get(catalog, [s], [s])
    unlock(rcf_cache_lock)
    entry, primary_rcf = ep[1]

    t_box = BoundingBox(entry.pos[1] - 1e-8, entry.pos[1] + 1e-8,
                        entry.pos[2] - 1e-8, entry.pos[2] + 1e-8)
    surrounding_rcfs = get_overlapping_fields(t_box, stagedir)

    local_images = Vector{TiledImage}()
    local_catalog = Vector{CatalogEntry}()
    for rcf in surrounding_rcfs
        lock(rcf_cache_lock)
        cached_images, cached_catalog, mem_handle = get!(rcf_cache, rcf) do
            n = rcf_to_index[rcf.run, rcf.camcol, rcf.field]
            @assert n > 0
            nputs(nodeid, "getting image $n for $(rcf.run), $(rcf.camcol), $(rcf.field)")
            fimgs, mem_handle = get(images, [n], [n])
            imgs = Vector{TiledImage}()
            for fimg in fimgs[1]
                img = Image(fimg)
                push!(imgs, TiledImage(img))
            end
            if n == 1
                s_a = 1
                st, st_handle = get(catalog_offset, [n], [n])
                s_b = st[1]
            else
                st, st_handle = get(catalog_offset, [n-1], [n])
                s_a = st[1]
                s_b = st[2]
            end
            catalog_entries, ce_handle = get(catalog, [s_a], [s_b])
            neighbors = [entry[1] for entry in catalog_entries]
            imgs, neighbors, mem_handle
        end
        unlock(rcf_cache_lock)
        append!(local_images, cached_images)
        append!(local_catalog, cached_catalog)
    end

    i = findfirst(local_catalog, entry)
    neighbor_indexes = Infer.find_neighbors([i,], local_catalog, local_images)[1]
    neighbors = local_catalog[neighbor_indexes]

    #nputs(nodeid, "starting inference for $s")
    t0 = time()
    vs_opt = Infer.infer_source(local_images, neighbors, entry)
    runtime = time() - t0
    #nputs(nodeid, "inference for $s complete")

    InferResult(entry.thing_id, entry.objid, entry.pos[1], entry.pos[2],
                vs_opt, runtime)
end


function optimize_sources(images, catalog, tasks, catalog_offset, task_offset,
            rcf_to_index, stagedir, results, timing)
    num_work_items = length(tasks)

    # cache for RCF data
    rcf_cache = Dict{RunCamcolField,
                     Tuple{Vector{TiledImage},
                           Vector{CatalogEntry},
                           GarrayMemoryHandle}}()
    rcf_cache_lock = SpinLock()

    # create Dtree and get the initial allocation
    dt, isparent = Dtree(num_work_items, 0.4,
                         ceil(Int64, Base.Threads.nthreads() / 2))
    numwi, (startwi, endwi) = initwork(dt)
    nputs(nodeid, "initially $numwi work items ($startwi-$endwi)")
    rundt = runtree(dt)
    workitems, wi_handle = get(tasks, [startwi], [endwi])
    widx = 1
    wilock = SpinLock()

    function process_tasks()
        tid = threadid()
        if rundt && tid == 1
            ntputs(nodeid, tid, "running tree")
            while runtree(dt)
                Garbo.cpu_pause()
            end
        else
            while true
                lock(wilock)
                if endwi == 0
                    ntputs(nodeid, tid, "out of work")
                    unlock(wilock)
                    break
                end
                if widx > numwi
                    ntputs(nodeid, tid, "consumed last work item; requesting more")
                    numwi, (startwi, endwi) = getwork(dt)
                    workitems, wi_handle = get(tasks, [startwi], [endwi])
                    widx = 1
                    ntputs(nodeid, tid, "got $numwi work items ($startwi-$endwi)")
                    unlock(wilock)
                    continue
                end
                taskidx = startwi+widx-1
                item = workitems[widx]
                widx = widx + 1
                unlock(wilock)
                ntputs(nodeid, tid, "processing source $item")

                result = optimize_source(item, images, catalog, catalog_offset,
                                rcf_to_index, rcf_cache, rcf_cache_lock,
                                stagedir, results)
                lock(rcf_cache_lock)
                put!(results, [taskidx], [taskidx], [result])
                unlock(rcf_cache_lock)
            end
        end
    end

    tic()
    ccall(:jl_threading_run, Void, (Any,), Core.svec(process_tasks))
    #process_tasks()
    #ccall(:jl_threading_profile, Void, ())
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

    if length(tasks) > 0
        # create map from run, camcol, field to index into RCF array
        rcf_to_index = invert_rcf_array(rcfs)

        # inference results are written here
        results = Garray(InferResult, 350, length(tasks))

        # optimization -- little disk access, cpu intensive
        timing.num_srcs = length(tasks)
        optimize_sources(images, catalog, tasks, catalog_offset, task_offset,
                rcf_to_index, stagedir, results, timing)

        rlo, rhi = distribution(results, nodeid)
        lresults = access(results, rlo, rhi)
        tic()
        save_results(outdir, box, lresults)
        timing.write_results = toq()

        finalize(results)
    end

    finalize(tasks)
    finalize(catalog)
    finalize(task_offset)
    finalize(catalog_offset)
    finalize(images)
end


