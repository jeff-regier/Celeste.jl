using FixedSizeArrays


###########  flatten catalog for use with Garbo #####

type FlatCatalogEntry
    pos::Vec{2, Float64}
    is_star::Bool
    star_fluxes::Vec{5, Float64}
    gal_fluxes::Vec{5, Float64}
    gal_frac_dev::Float64
    gal_ab::Float64
    gal_angle::Float64
    gal_scale::Float64
    objid::Vec{19, UInt8}
    thing_id::Int
end


function convert(::Type{FlatCatalogEntry}, ce::CatalogEntry)
    @assert length(ce.objid) == 19
    FlatCatalogEntry(
        ce.pos,
        ce.is_star,
        ce.star_fluxes,
        ce.gal_fluxes,
        ce.gal_frac_dev,
        ce.gal_ab,
        ce.gal_angle,
        ce.gal_scale,
        convert(Vector{UInt8}, ce.objid),
        ce.thing_id)
end


function convert(::Type{CatalogEntry}, ce::FlatCatalogEntry)
    CatalogEntry(
        ce.pos,
        ce.is_star,
        ce.star_fluxes,
        ce.gal_fluxes,
        ce.gal_frac_dev,
        ce.gal_ab,
        ce.gal_angle,
        ce.gal_scale,
        convert(Vector{UInt8}, ce.objid),
        ce.thing_id)
end


###########  flatten tiled image for use with Garbo #####

immutable FlatTiledImage
    # The image height.
    H::Int

    # The image width.
    W::Int

    # subimages
    tiles::Mat{100,200,ImageTile}

    # all tiles have the same height and width
    tile_width::Int

    # The band id (takes on values from 1 to 5).
    b::Int

    # World coordinates
    wcs_header::Vec{10000,UInt8}

    # The components of the point spread function.
    psf::Vec{psf_K, PsfComponent}

    # SDSS-specific identifiers. A field is a particular region of the sky.
    # A Camcol is the output of one camera column as part of a Run.
    run_num::Int
    camcol_num::Int
    field_num::Int

    # storing a RawPSF here isn't ideal, because it's an SDSS type
    # not a Celeste type
    raw_psf_comp::RawPSF
end


function convert(::Type{FlatTiledImage}, img::TiledImage)
    wcs_header = WCS.to_header(img.wcs)
    # Kiran, I think the wcs_header will always be shorter than 
    # 10000 characters
    @assert(length(wcs_header) < 10_000)
    FlatTiledImage(img.H, img.W, img.tiles, img.tile_width, img.b, wcs_header,
                   img.psf, img.run_num, img.camcol_num, img.field_num,
                   img.raw_psf_comp)
end


function convert(::Type{TiledImage}, img::FlatTiledImage)
    wcs_array = WCS.from_header(img.wcs_header)
    @assert(length(wcs_array) == 1)
    wcs = wcs_array[1]
    TiledImage(img.H, img.W, img.tiles, img.tile_width, img.b, wcs,
               img.psf, img.run_num, img.camcol_num, img.field_num,
               img.raw_psf_comp)
end


###########################################################


function fetch_catalog(rcf, stagedir)
    # note: this call to read_photoobj_files considers only primary detections.
    catalog = SDSSIO.read_photoobj_files([rcf,], stagedir)

    # we're ignoring really faint sources entirely...not even using them to
    # render the background
    isnt_faint(entry) = maximum(entry.star_fluxes) >= MIN_FLUX
    filter(isnt_faint, catalog)
end


function in_box(entry::CatalogEntry, box::BoundingBox)
    box.ramin < entry.pos[1] < box.ramax &&
        box.decmin < entry.pos[2] < box.decmax
end


function load_images(box, rcfs, stagedir)
    num_fields = length(rcfs)

    # each cell of `images` contains B=5 tiled images
    images = Garray(NTuple{5,FlatTiledImage}, num_fields)

    # stores first index of each field's sources in the catalog array
    catalog_offset = Garray(Int64, num_fields)

    # stores first index of each field's tasks in the tasks array
    task_offset = Garray(Int64, num_fields)

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
        raw_images = SDSSIO.load_field_images(rcf, stagedir)
        @assert(length(raw_images) == 5)
        timgs = [convert(FlatTiledImage, TiledImage(img)) for img in raw_images]
        sizeof(timgs[1])
        limages[i] = tuple(timgs...)
   
        # second, load the `catalog_offset` and `task_count` arrays with
        # a number of sources for each field.
        # (We'll accumulate the entries later.)
        local_catalog = fetch_catalog(rcf, stagedir)
        cef = convert(FlatCatalogEntry, local_catalog[1])
        println(sizeof(cef))

        # we'll use sources outside of the box to render the background,
        # but we won't optimize them
        local_tasks = filter(s->in_box(s, box), local_catalog)

        lcatalog_offset[i] = length(local_catalog)
        ltask_offset[i] = length(local_tasks)
    end
    Garbo.flush(images)
    Garbo.flush(catalog_offset)
    Garbo.flush(task_offset)
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
        Garbo.flush(catalog_offset)
        Garbo.flush(task_offset)
        sync()
    end

    images, catalog_offset, task_offset
end


function load_catalog(box, rcfs, catalog_offset, task_offset, stagedir)
    num_fields = length(rcfs)
    catalog_size = catalog_offset[end]
    num_tasks = task_offset[end]

    catalog = Garray(Tuple{FlatCatalogEntry, RunCamcolField}, catalog_size)

    # entries in `tasks` are indexes into `catalog`
    tasks = Garray(Int64, num_tasks)

    # get local distribution of the arrays
    colo, cohi = distribution(catalog_offset, nodeid)
    nlocal = cohi[1]-colo[1]+1
    clo, chi = distribution(catalog, nodeid)
    lcatalog_size = chi[1]-clo[1]+1
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
        max_run = maximum(max_run, rcf.run)
        max_camcol = maximum(max_run, rcf.camcol)
        max_field = maximum(max_run, rcf.field)
    end

    rcf_to_index = zeros(Int64, max_run, max_camcol, max_field)

    # this should be really fast, each node could do it
    for n in 1:length(rcfs)
        rcf = rcfs[n]
        rcf_to_index[rcf.run, rcf.camcol, rcf.field] = n
    end

    rcf_to_index
end


function optimize_source(s, catalog, catalog_offset, rcf_to_index, images)
    local_images = Vector{TiledImage}[]
    local_catalog = CatalogEntry[];

    entry, primary_rcf = catalog[s]
    t_box = BoundingBox(entry.pos[1] - 1e-8, entry.pos[1] + 1e-8,
                        entry.pos[2] - 1e-8, entry.pos[2] + 1e-8)
    surrounding_rcfs = get_overlapping_fields(t_box, stagedir)

    for rcf in surrounding_rcfs
        n = rcf_to_index[rcf.run, rcf.camcol, rcf.field]

        push!(local_images, images[n])

        s_a = n == 1 ? 1 : catalog_offset[n - 1] + 1
        s_b = catalog_offset[n]
        for neighbor in catalog[s_a:s_b]
            push!(local_catalog, neighbor[1])
        end
    end

    flat_images = TiledImage[]
    for img5 in local_images, img in img5
        push!(flat_images, img)
    end

    i = findfirst(local_catalog, entry)
    neighbor_indexes = Infer.find_neighbors([i,], local_catalog, flat_images)[1]
    neighbors = local_catalog[neighbor_indexes]

    vs_opt = Infer.infer_source(flat_images, neighbors, entry)
    # TODO: write vs_opt (the results) to disk, to a global array of size num_tasks,
    # or to a local array, and then flush the array to disk later.

    push!(results, Dict(
        "thing_id"=>entry.thing_id,
        "objid"=>entry.objid,
        "ra"=>entry.pos[1],
        "dec"=>entry.pos[2],
        "vs"=>vs_opt))
end


function optimize_sources(images, catalog, tasks,
                          catalog_offset, task_offset, 
                          rcf_to_index, stagedir)
    #TODO: set `nnodes` to the number of nodes, and
    #`nthreads` to the number of threads per node.
    num_nodes = 2
    threads_per_node = 3
    total_threads = num_nodes * threads_per_node

    num_tasks = task_offset[end]

    tasks_per_thread = ceil(Int, num_tasks / total_threads)

    results = Dict[]

    for m in 0:(total_threads-1)
        t0 = 1 + m * tasks_per_thread
        tend = min(num_tasks, (m + 1) * tasks_per_thread)

        for t in t0:tend
            s = tasks[t]
            entry, primary_rcf = catalog[s]
            t_box = BoundingBox(entry.pos[1] - 1e-8, entry.pos[1] + 1e-8,
                                entry.pos[2] - 1e-8, entry.pos[2] + 1e-8)
            surrounding_rcfs = get_overlapping_fields(t_box, stagedir)
            local_images = Vector{TiledImage}[]
            local_catalog = CatalogEntry[];

            for rcf in surrounding_rcfs
                n = rcf_to_index[rcf.run, rcf.camcol, rcf.field]

                push!(local_images, images[n])

                s_a = n == 1 ? 1 : catalog_offset[n - 1] + 1
                s_b = catalog_offset[n]
                for neighbor in catalog[s_a:s_b]
                    push!(local_catalog, neighbor[1])
                end
            end

            flat_images = TiledImage[]
            for img5 in local_images, img in img5
                push!(flat_images, img)
            end

            i = findfirst(local_catalog, entry)
            neighbor_indexes = Infer.find_neighbors([i,], local_catalog, flat_images)[1]
            neighbors = local_catalog[neighbor_indexes]

            vs_opt = Infer.infer_source(flat_images, neighbors, entry)
            # TODO: write vs_opt (the results) to disk, to a global array of size num_tasks,
            # or to a local array, and then flush the array to disk later.

            push!(results, Dict(
                "thing_id"=>entry.thing_id,
                "objid"=>entry.objid,
                "ra"=>entry.pos[1],
                "dec"=>entry.pos[2],
                "vs"=>vs_opt))
        end
    end

    results
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
    rcfs = get_overlapping_fields(box, stagedir)

    # loads 25TB from disk for SDSS
    images, catalog_offset, task_offset = load_images(box, rcfs, stagedir)

    # loads 4TB from disk for SDSS
    catalog, tasks = load_catalog(box, rcfs, catalog_offset, task_offset, stagedir)

    # create map from run, camcol, field to index into RCF array
    rcf_to_index = invert_rcf_array(rcfs)

    # optimization -- little disk access, cpu intensive
    results = optimize_sources(images, catalog, tasks,
                               catalog_offset, task_offset,
                               rcf_to_index, stagedir)
end


