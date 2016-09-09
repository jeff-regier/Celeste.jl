using StaticArrays
import WCS


###########  flattened data types for use with Garbo #####

type FlatCatalogEntry
    pos::SVector{2,Float64}
    is_star::Bool
    star_fluxes::SVector{5,Float64}
    gal_fluxes::SVector{5,Float64}
    gal_frac_dev::Float64
    gal_ab::Float64
    gal_angle::Float64
    gal_scale::Float64
    objid::SVector{19,UInt8}
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

####

immutable FlatRawPSF
    rrows::SMatrix{2601,4,Float64}  # A matrix of flattened eigenimages.
    rnrow::Int  # The number of rows in an eigenimage.
    rncol::Int  # The number of columns in an eigenimage.
    cmat::SVector{4,SMatrix{5,5,Float64}}  # The coefficients of the weight polynomial
    nrow_b::Int
    ncol_b::Int

    function FlatRawPSF(rrows::Array{Float64, 2}, rnrow::Integer, rncol::Integer,
                     cmat_raw::Array{Float64, 3}, nrow_b::Integer, ncol_b::Integer)
        # rrows contains eigen images. Each eigen image is along the first
        # dimension in a flattened form. Check that dimensions match up.
        @assert size(rrows, 1) == rnrow * rncol

        # The second dimension is the number of eigen images, which should
        # match the number of coefficient arrays.
        @assert size(rrows, 2) == size(cmat_raw, 3)

        cmat2 = Matrix[cmat_raw[:,:,i] for i in 1:size(cmat_raw, 3)]

        return new(rrows, Int(rnrow), Int(rncol), cmat2, Int(nrow_b), Int(ncol_b))
    end
end

function convert(::Type{RawPSF}, psf::FlatRawPSF)
    RawPSF(psf.rrows, psf.rnrow, psf.rncol, psf.cmat[1:psf.nrow_b, 1:psf.ncol_b])
end

function convert(::Type{FlatRawPSF}, psf::RawPSF)
    cmat = zeros(5, 5, 4)
    for k in 1:size(psf.cmat, 3)
        cmat[size(psf.cmat, 1), size(psf.cmat, 2), k] = cmat[:, :, k]
    end
    FlatRawPSF(psf.rrows, psf.rnrow, psf.rncol, cmat, psf.nrow_b, psf.ncol_b)
end

####

immutable FlatPsfComponent
    alphaBar::Float64
    xiBar::SVector{2,Float64}
    tauBar::SMatrix{2,2,Float64}
    tauBarInv::SMatrix{2,2,Float64}
    tauBarLd::Float64
end

function convert(::Type{FlatPsfComponent}, c::PsfComponent)
    FlatPsfComponent(c.alphaBar. c.xiBar, c.tauBar, c.tauBarInv, c.tauBarLd)
end

function convert(::Type{PsfComponent}, c::FlatPsfComponent)
    PsfComponent(c.alphaBar. c.xiBar, c.tauBar, c.tauBarInv, c.tauBarLd)
end

####

immutable FlatImageTile
    b::Int
    h_range::UnitRange{Int}
    w_range::UnitRange{Int}
    pixels::SMatrix{20, 20, Float32}
    epsilon_mat::SMatrix{20, 20, Float32}
    iota_vec::SVector{20, Float32}
end

function convert(::Type{FlatImageTile}, it::ImageTile)
    FlatImageTile(it.b, it.h_range, it.w_range, it.pixels, it.epsilon_mat, it.iota_vec)
end

function convert(::Type{ImageTile}, it::FlatImageTile)
    ImageTile(it.b, it.h_range, it.w_range, it.pixels, it.epsilon_mat, it.iota_vec)
end

####

immutable FlatTiledImage
    H::Int
    W::Int
    tiles::SMatrix{100,200,FlatImageTile}
    tile_width::Int
    b::Int
    wcs_header::SVector{10000,UInt8}
    psf::SVector{psf_K,FlatPsfComponent}
    run_num::Int
    camcol_num::Int
    field_num::Int
    raw_psf_comp::FlatRawPSF
end

function convert(::Type{FlatTiledImage}, img::TiledImage)
    wcs_header = WCS.to_header(img.wcs)
    @assert(length(wcs_header) < 10_000)

    n = length(wcs_header)
    wcs_header_bytes = zeros(UInt8, 10_000)
    for i in 1:n
        wcs_header_bytes[i] = wcs_header[i]
    end

    FlatTiledImage(img.H, img.W, img.tiles, img.tile_width, img.b, wcs_header_bytes,
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


###########  flat type to store inference results #########

immutable InferResult
    thing_id::Int
    objid::SVector{30,UInt8}
    ra::Float64
    dec::Float64
    vs::SVector{88,Float64}
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


function optimize_source(s, images, catalog, catalog_offset, rcf_to_index, results)
    local_images = Vector{TiledImage}[]
    local_catalog = CatalogEntry[];

    ep = get(catalog, [s], [s])
    entry, primary_rcf = ep[1]
    t_box = BoundingBox(entry.pos[1] - 1e-8, entry.pos[1] + 1e-8,
                        entry.pos[2] - 1e-8, entry.pos[2] + 1e-8)
    surrounding_rcfs = get_overlapping_fields(t_box, stagedir)

    for rcf in surrounding_rcfs
        n = rcf_to_index[rcf.run, rcf.camcol, rcf.field]
        append!(local_images, get(images, [n], [n]))
        if n == 1
            s_a = 1
            st = get(catalog_offset, [n], [n])
            s_b = st[1]
        else
            st = get(catalog_offset, [n-1], [n])
            s_a = st[1]
            s_b = st[2]
        end
        neighbors = get(catalog, [s_a], [s_b])
        for neighbor in neighbors
            push!(local_catalog, neighbor[1])
        end
    end

    flat_images = [img for img5 in local_images for img in img5]

    i = findfirst(local_catalog, entry)
    neighbor_indexes = Infer.find_neighbors([i,], local_catalog, flat_images)[1]
    neighbors = local_catalog[neighbor_indexes]

    vs_opt = Infer.infer_source(flat_images, neighbors, entry)

    put!(results, [s], [s], [InferResult(entry.thing_id, entry.objid,
                                    entry.pos[1], entry.pos[2], vs_opt)])
end


function process_tasks(dt, rundt, wi, wilock, ci, li,
                images, catalog, tasks, catalog_offset, task_offset,
                rcf_to_index, stagedir, results)
    tid == threadid()
    if rundt && tid == 1
        ntputs(nodeid, tid, "running tree")
        while runtree(dt)
            cpu_pause()
        end
    else
        while true
            lock(wilock)
            if li[] == 0
                ntputs(nodeid, tid, "out of work")
                unlock(wilock)
                break
            end
            if ci[] == li[]
                ntputs(nodeid, tid, "consumed last work item ($(li[])); requesting more")
                ni, (ci, li) = getwork(dt)
                wi = get(tasks, [ci], [li])
                ci[] = 1
                li[] = ni
                ntputs(nodeid, tid, "got $ni work items")
                unlock(wilock)
                continue
            end
            item = ci[]
            ci[] = ci[] + 1
            unlock(wilock)

            optimize_source(item, images, catalog, catalog_offset, rcf_to_index, results)
        end
    end
end


function optimize_sources(images, catalog, tasks, catalog_offset, task_offset,
            rcf_to_index, stagedir, results, timing)
    num_work_items = length(tasks)
    if nodeid == 1
        nputs(nodeid, "processing $num_work_items work items on $nnodes nodes")
    end

    # create Dtree and get the initial allocation
    dt, isparent = DtreeScheduler(num_work_items, 0.4)
    ni, (ci, li) = initwork(dt)
    wi = get(tasks, [ci], [li])
    ci = 1
    li = ni
    wilock = SpinLock()

    tic()
    ptargs = Core.svec(process_tasks, dt, runtree(dt), wi, wilock,
                    Ref(ci), Ref(li),
                    images, catalog, tasks, catalog_offset, task_offset,
                    rcf_to_index, stagedir, results)
    ccall(:jl_threading_run, Void, (Any,), ptargs)
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
    results = Garray(InferResult, length(catalog))

    # optimization -- little disk access, cpu intensive
    timing.num_srcs = length(tasks)
    optimize_sources(images, catalog, tasks, catalog_offset, task_offset,
            rcf_to_index, stagedir, results, timing)
end


