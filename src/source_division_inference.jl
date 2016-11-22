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


function invert_rcf_array(rcfs)
    max_run = maximum([rcf.run for rcf in rcfs])
    max_camcol = maximum([rcf.camcol for rcf in rcfs])
    max_field = maximum([rcf.field for rcf in rcfs])

    rcf_to_index = Array(Int64, max_run, max_camcol, max_field)

    fill!(rcf_to_index, -1)

    # this should be really fast, each node could do it
    for n in 1:length(rcfs)
        rcf = rcfs[n]
        rcf_to_index[rcf.run, rcf.camcol, rcf.field] = n
    end

    rcf_to_index
end


function load_images(box, rcfs, stagedir)
    # TODO: set `total_threads`. We may not want to use all threads possible
    # on all nodes, since this operation hits disk. Maybe just 1 thread per node?
    total_threads = 5
    num_fields = length(rcfs)
    fields_per_thread = ceil(Int, num_fields / total_threads)

    #TODO: make `images` a global array
    # (each cell of `images` contains B=5 images)
    images = Array(Vector{Image}, num_fields)

    #TODO: make `catalog_offset` a global array too.
    # It stores first index of each field's sources in the catalog array.
    catalog_offset = Array(Int64, num_fields)

    #TODO: make `task_offset` a global array too.
    # It stores first index of each field's tasks in the tasks array.
    task_offset = Array(Int64, num_fields)

    #TODO: use dtree to have each thread on each node
    #  load some fields.
    #Note: this loop and the next one over threads hit disk, so we may not want
    #  to use all available threads. (The third loop over threads is the 
    #  only cpu intensive loop.)
    for m in 0:(total_threads - 1)
        n0 = 1 + m * fields_per_thread
        nend = min(num_fields, (m + 1) * fields_per_thread)

        for n in n0:nend
            # first, load the field array
            rcf = rcfs[n]
            images[n] = SDSSIO.load_field_images(rcf, stagedir)
            @assert(length(images[n]) == 5)
   
            # second, load the `catalog_offset` and `task_count` arrays with
            # a number of sources for each field.
            # (We'll accumulate the entries later.)
            local_catalog = fetch_catalog(rcf, stagedir)

            # we'll use sources outside of the box to render the background,
            # but we won't optimize them
            local_tasks = filter(s->in_box(s, box), local_catalog)

            catalog_offset[n] = length(local_catalog)
            task_offset[n] = length(local_tasks)
        end
    end

   # folds right, converting each field's count to its offset in `catalog`
   catalog_size = 0
    for n in 1:num_fields
        catalog_size += catalog_offset[n]
        catalog_offset[n] = catalog_size
    end

    # folds right, converting each field's count to its offset in `tasks`
    num_tasks = 0
    for n in 1:num_fields
        num_tasks += task_offset[n]
        task_offset[n] = num_tasks
    end

    images, catalog_offset, task_offset
end


function load_catalog(box, rcfs, catalog_offset, task_offset, stagedir)
    # TODO: set `total_threads`. We may not want to use all threads possible
    # on all nodes, since this operation hits disk. Maybe just 1 thread per node?
    total_threads = 3
    num_fields = length(rcfs)
    fields_per_thread = ceil(Int, num_fields / total_threads)

    catalog_size = catalog_offset[end]
    num_tasks = task_offset[end]

    #TODO: make the `catalog` a GlobalArray
    catalog = Array(Tuple{CatalogEntry, RunCamcolField}, catalog_size)

    # entries in `tasks` are indexes into `catalog`
    #TODO: make the `tasks` a GlobalArray
    tasks = Array(Int64, num_tasks)

    #let's read the catalog from disk again, rather than storing it in memory
    for m in 0:(total_threads-1)
        n0 = 1 + m * fields_per_thread
        nend = min(num_fields, (m + 1) * fields_per_thread)

        for n in n0:nend
            rcf = rcfs[n]

            local_catalog = fetch_catalog(rcf, stagedir)

            local_t = 0
            for local_s in 1:length(local_catalog)
                # `s` is the global index of this source (in the source array)
                offset = n == 1 ? 0 : catalog_offset[n - 1]
                s = offset + local_s
                # Note: each field's primary detections are stored contiguously
                # in the source array
                entry = local_catalog[local_s]
                catalog[s] = (entry, rcf)

                if in_box(entry, box)
                    local_t += 1
                    offset = n == 1 ? 0 : task_offset[n - 1]
                    t = offset + local_t
                    tasks[t] = s
                end
            end
        end
    end

    catalog, tasks
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
            local_images = Vector{Image}[]
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

            flat_images = Image[]
            for img5 in local_images, img in img5
                push!(flat_images, img)
            end

            i = findfirst(local_catalog, entry)
            neighbor_indexes = Infer.find_neighbors([i,], local_catalog, flat_images)[1]
            neighbors = local_catalog[neighbor_indexes]

            vs_opt = infer_source(flat_images, neighbors, entry)
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

    # mulithreaded operation 1 (of 3) -- loads 25TB from disk for SDSS
    images, catalog_offset, task_offset = load_images(box, rcfs, stagedir)

    # mulithreaded operation 2 (of 3) -- loads 4TB from disk for SDSS
    catalog, tasks = load_catalog(box, rcfs, catalog_offset, task_offset, stagedir)

    rcf_to_index = invert_rcf_array(rcfs)

    # mulithreaded operation 3 (of 3) -- little disk access, cpu intensive
    results = optimize_sources(images, catalog, tasks,
                               catalog_offset, task_offset,
                               rcf_to_index, stagedir)
end


