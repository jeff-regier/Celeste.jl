function load_images(rcfs, stagedir)
    images = TiledImage[]
    image_names = String[]
    image_count = 0

    for i in 1:length(rcfs)
        Log.info("reading field $(rcfs[i])")
        rcf = rcfs[i]
    end
    gc()

    Log.debug("Image names:")
    Log.debug(string(image_names))

    images
end



"""
Fit the Celeste model to sources in a given ra, dec range,
based on data from specified fields
- box: a bounding box specifying a region of sky
"""
function divide_sources_and_infer(
                box::BoundingBox,
                stagedir::String,
                timing=InferTiming(),
                outdir=".")
    # read the run-camcol-field triplets for this box
    rcfs = get_overlapping_fields(box, stagedir)

    #TODO: make `field_array` a global array
    # (each cell of `field_array` contains B=5 tiled images)
    N = length(rcfs)
    field_array = Array(Vector{TiledImage}, N)

    #TODO: set `nnodes` to the number of nodes, and
    #`nthreads` to the number of threads per node.
    nnodes = 2
    nthreads = 3
    M = nnodes * nthreads
    fields_per_thread = ceil(Int, N / M)

    #TODO: make `field_source_offset` a global array too.
    # It stores first index of each field's sources in the source array.
    field_source_offset = Array(Int64, N)

    #TODO: use dtree to have each thread on each node
    #load some fields, e.g.,
    for m in 0:(M-1)
        first_index = 1 + m * fields_per_thread
        last_index = (m + 1) * fields_per_thread
        for n in first_index:last_index
            # first, load the field array
            rcf = rcfs[n]
            raw_images = SDSSIO.load_field_images(rcf, stagedir)
            field_array[n] = [TiledImage(img) for img in raw_field_images]
   
            # second, load the `field_source_offset` array with
            # the number of sources in each field.
            # (We'll accumulate these entries later.)
            # Note: this call to read_photoobj_files loads only primary detections.
            rcf_catalog = SDSSIO.read_photoobj_files([rcf,], stagedir)
            rcf_catalog = filter(entry->(maximum(entry.star_fluxes) >= MIN_FLUX), catalog)
            field_source_count[n] = length(rcf_catalog)
        end
    end

    # count the number of sources S and convert our per field counts to
    # source-array offsets
    S = 0
    for n in 1:N
        S += field_source_offset[n]
        field_source_offset[n] = S
    end

    #TODO: make the `source_array` a GlobalArray
    source_array = Array(Tuple{CatalogEntry, FieldTriplet}, S)

    #let's read the catalog from disk again, rather than storing it in memory
    for m in 0:(M-1)
        first_index = 1 + m * fields_per_thread
        last_index = (m + 1) * fields_per_thread
        for n in first_index:last_index
            rcf_catalog = SDSSIO.read_photoobj_files([rcf,], stagedir)
            rcf_catalog = filter(entry->(maximum(entry.star_fluxes) >= MIN_FLUX), catalog)
            for s1 in 1:length(rcf_catalog)
                # `s` is the global index of this source (in the source array)
                s = field_source_offset[n] + s1
                source_array[s] = (rcf_catalog[s1], rcf)
            end
        end
    end

    sources_per_thread = ceil(Int, S / M)

    max_run = maximum([rcf.run for rcf in rcfs])
    max_camcol = maximum([rcf.camcol for rcf in rcfs])
    max_field = maximum([rcf.field for rcf in rcfs])
    image_map = Array(Int64, max_run, max_camcol, max_field)
    fill!(image_map, 0)

    # TODO: make `images` a 1D GlobalArray
    images = load_images(rcfs, stagedir)

    neighbor_map = Infer.find_neighbors(target_sources, catalog, images)
    # let's process some sources!!!
    for m in 0:(M-1)
        s_a = 1 + m * sources_per_thread
        s_b = (m + 1) * sources_per_thread
        for s in s_a:s_b
            entry, rcf = source_array[s]
            neightbors = catalog[neighbor_map[ts]]
            vs_opt = Infer.infer_source(images, neighbors, entry)
            # TODO: write vs_opt (the results) to disk, to a global array of size S,
            # or to a local array, and then flush the array to disk later.
        end
    end
end


