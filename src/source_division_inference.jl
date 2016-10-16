using Base.Threads
using StaticArrays

import WCS
import Base.convert, Base.serialize, Base.deserialize

# Serialized type sizes:
# - CatalogEntry size is 202 (pos:2, star_fluxes:5, gal_fluxes:5, objid:19)
# - RawPSF size is 84861 (rrows: 2601x4, cmat: 4x5x5)
# - PsfComponent size is 130 (xiBar: 2, tauBar:2x2, tauBarInv:2x2)
# - FlatImage size is 24499147 (pixels: 2048x1489, epsilon_mat: 1489, iota_vec: 2048)
# - NTuple{5,FlatImage} is 122495737
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
    @assert(length(wcs_header) <= 10000)
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
    if length(a) > flen
        nputs(nodeid, "error: ser_array dims are $(size(a)), > $flen")
    end
    @assert(length(a) <= flen)
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
    for p in psf.xiBar
        write(s.io, p)
    end
    for p in psf.tauBar
        write(s.io, p)
    end
    for p in psf.tauBarInv
        write(s.io, p)
    end
    #ser_array(s, psf.xiBar, 2)
    #ser_array(s, psf.tauBar, 4)
    #ser_array(s, psf.tauBarInv, 4)
    write(s.io, psf.tauBarLd)
end

function deserialize(s::Base.AbstractSerializer, t::Type{PsfComponent})
    alphaBar = read(s.io, Float64)::Float64
    x = zeros(Float64, 2)
    for i = 1:length(x)
        x[i] = read(s.io, Float64)::Float64
    end
    xiBar = SVector{2}(x)
    x = zeros(Float64, 2, 2)
    for i = 1:length(x)
        x[i] = read(s.io, Float64)::Float64
    end
    tauBar = SMatrix{2,2}(x)
    x = zeros(Float64, 2, 2)
    for i = 1:length(x)
        x[i] = read(s.io, Float64)::Float64
    end
    tauBarInv = SMatrix{2,2}(x)
    #xiBar = deser_array(s, Float64, 2)
    #tauBar = deser_array(s, Float64, 4)
    #tauBarInv = deser_array(s, Float64, 4)
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
    @assert(whlen <= 10000)
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

        raw_images = SDSSIO.load_field_images(rcf, stagedir)
        @assert(length(raw_images) == 5)
        fimgs = [FlatImage(img) for img in raw_images]
        limages[i] = tuple(fimgs...)
   
        # load the catalog entries for this RCF
        local_catalog = fetch_catalog(rcf, stagedir)

        # we'll use sources outside of the box to render the background,
        # but we won't optimize them
        local_tasks = filter(s->in_box(s, box), local_catalog)

        nputs(nodeid,
              "$(length(local_catalog)) sources ($(length(local_tasks)) tasks)",
              " in RCF $n ($(rcf.run), $(rcf.camcol), $(rcf.field))")

        # only store the number of catalog entries and tasks
        lcatalog_offset[i] = length(local_catalog)
        ltask_offset[i] = length(local_tasks)
    end
    flush(images)
    flush(catalog_offset)
    flush(task_offset)
    sync()

    lcatalog_offset = access(catalog_offset, lo, hi)
    ltask_offset = access(task_offset, lo, hi)

    # folds right, converting each field's count to offsets in `catalog`
    # and `tasks`. this is a prefix sum, which can be done in parallel
    # (TODO: add this to Garbo) but is being done sequentially here
    for nid = 1:nnodes
        # one node at a time
        if nid == nodeid && nlocal > 0
            catalog_size = 0
            num_tasks = 0
            if nid > 1
                coa, ch = get(catalog_offset, lo-1, lo-1)
                #nputs(nodeid, "got $(coa[1]) from $(lo[1]-1)")
                catalog_size = catalog_size + coa[1]
                toa, th = get(task_offset, lo-1, lo-1)
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
    coa, ch = get(catalog_offset, [num_fields], [num_fields])
    catalog_size = coa[1]
    toa, th = get(task_offset, [num_fields], [num_fields])
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
    cat_idx = 1
    task_idx = 1
    colo, cohi = distribution(catalog_offset, nodeid)
    if colo[1] > 1
        coa, ch = get(catalog_offset, [colo[1]-1], [colo[1]-1])
        cat_idx = cat_idx + coa[1]
        toa, th = get(task_offset, [colo[1]-1], [colo[1]-1])
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
    sync()

    catalog, tasks
end


"""
Build an index to quickly map an RCF to its index in the set of RCFs
provided.
"""
function invert_rcf_array(rcfs::Vector{RunCamcolField})
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


"""
Evicts the LRU entry in the RCF cache.
"""
function clean_cache(cache::Dict)
    lru_rcf = RunCamcolField(0, 0, 0)
    lru_interval = 0.0 
    curr_time = time()
    for (rcf, (imgs, ih, cat, ch, lru)) in cache
        use_interval = curr_time - lru 
        if use_interval > lru_interval
            lru_rcf = rcf 
            lru_interval = use_interval
        end 
    end 
    ntputs(nodeid, threadid(), "discarding $(lru_rcf.run), $(lru_rcf.camcol), $(lru_rcf.field)")
    delete!(cache, lru_rcf)
end


"""
Sets up for, and runs inference on the specified source. Thread-safe.
"""
function optimize_source(s::Int64, images::Garray, catalog::Garray,
                         catalog_offset::Garray, rcf_to_index::Array{Int64,3},
                         cache::Dict, cache_lock::SpinLock, g_lock::SpinLock,
                         stagedir::String, times::InferTiming)
    tid = threadid()

    tic()
    lock(g_lock)
    ep, eph = get(catalog, [s], [s])
    unlock(g_lock)
    times.ga_get = times.ga_get + toq()
    entry, primary_rcf = ep[1]

    t_box = BoundingBox(entry.pos[1] - 1e-8, entry.pos[1] + 1e-8,
                        entry.pos[2] - 1e-8, entry.pos[2] + 1e-8)
    surrounding_rcfs = get_overlapping_fields(t_box, stagedir)

    rimages = Vector{TiledImage}()
    rcatalog = Vector{CatalogEntry}()
    #tic()
    for rcf in surrounding_rcfs
        lock(cache_lock)
        cached_imgs, ih, cached_cat, ch, _ = get(cache, rcf) do
            n = rcf_to_index[rcf.run, rcf.camcol, rcf.field]
            @assert n > 0

            ntputs(nodeid, tid, "fetching RCF $n",
                   " ($(rcf.run), $(rcf.camcol), $(rcf.field))")

            tic()
            lock(g_lock)
            fimgs, ih = get(images, [n], [n])
            unlock(g_lock)
            times.ga_get = times.ga_get + toq()
            imgs = [TiledImage(Image(fimg)) for fimg in fimgs[1]]

            if n == 1
                s_a = 1
                tic()
                lock(g_lock)
                st, st_handle = get(catalog_offset, [n], [n])
                unlock(g_lock)
                times.ga_get = times.ga_get + toq()
                s_b = max(1, st[1])
            else
                tic()
                lock(g_lock)
                st, st_handle = get(catalog_offset, [n-1], [n])
                unlock(g_lock)
                times.ga_get = times.ga_get + toq()
                s_a = max(1, st[1])
                s_b = max(1, st[2])
            end
            tic()
            lock(g_lock)
            cat_entries, ch = get(catalog, [s_a], [s_b])
            unlock(g_lock)
            times.ga_get = times.ga_get + toq()
            neighbors = [ce[1] for ce in cat_entries]

            imgs, ih, neighbors, ch, time()
        end
        push!(cache, rcf => (cached_imgs, ih, cached_cat, ch, time()))
        if length(cache) > 20
            clean_cache(cache)
        end
        unlock(cache_lock)
        append!(rimages, cached_imgs)
        append!(rcatalog, cached_cat)
    end
    #ntputs(nodeid, tid, "fetched data to infer $s in $(toq()) secs")

    #tic()
    i = findfirst(rcatalog, entry)
    neighbor_indexes = Infer.find_neighbors([i,], rcatalog, rimages)[1]
    neighbors = rcatalog[neighbor_indexes]
    #ntputs(nodeid, tid, "loaded neighbors of $s in $(toq()) secs")

    t0 = time()
    vs_opt = Infer.infer_source(rimages, neighbors, entry)
    runtime = time() - t0
    rds = @sprintf "%s: %5.3f secs" entry.objid runtime
    ntputs(nodeid, tid, rds)

    InferResult(entry.thing_id, entry.objid, entry.pos[1], entry.pos[2],
                vs_opt, runtime)
end


function optimize_sources(images::Garray, catalog::Garray, tasks::Garray,
                          catalog_offset::Garray, task_offset::Garray,
                          rcf_to_index::Array{Int64,3}, stagedir::String,
                          timing::InferTiming)
    num_work_items = length(tasks)

    # inference results
    results = Vector{InferResult}()
    results_lock = SpinLock()

    # cache for RCF data; key is RCF, 
    cache = Dict{RunCamcolField,
                 Tuple{Vector{TiledImage},
                       GarrayMemoryHandle,
                       Vector{CatalogEntry},
                       GarrayMemoryHandle,
                       Float64}}()
    cache_lock = SpinLock()

    # to serialize get() calls
    g_lock = SpinLock()

    # per-thread timing
    ttimes = Array(InferTiming, nthreads())

    # create Dtree and get the initial allocation
    dt, isparent = Dtree(num_work_items, 0.4,
                         ceil(Int64, nthreads() / 4))
    numwi, (startwi, endwi) = initwork(dt)
    rundt = runtree(dt)

    nputs(nodeid, "dtree: initial work: $numwi ($startwi-$endwi)")

    workitems, wih = get(tasks, [startwi], [endwi])
    widx = 1
    wilock = SpinLock()

    function process_tasks()
        tid = threadid()
        ttimes[tid] = InferTiming()
        times = ttimes[tid]

        if rundt && tid == 1
            ntputs(nodeid, tid, "dtree: running tree")
            while runtree(dt)
                Garbo.cpu_pause()
            end
        else
            while true
                tic()
                lock(wilock)
                if endwi == 0
                    ntputs(nodeid, tid, "dtree: out of work")
                    unlock(wilock)
                    times.sched_ovh = times.sched_ovh + toq()
                    break
                end
                if widx > numwi
                    ntputs(nodeid, tid, "dtree: getting work")
                    lock(g_lock)
                    numwi, (startwi, endwi) = getwork(dt)
                    unlock(g_lock)
                    times.sched_ovh = times.sched_ovh + toq()
                    ntputs(nodeid, tid, "dtree: $numwi work items ($startwi-$endwi)")
                    if endwi > 0
                        tic()
                        lock(g_lock)
                        workitems, wih = get(tasks, [startwi], [endwi])
                        unlock(g_lock)
                        times.ga_get = times.ga_get + toq()
                        widx = 1
                    end
                    unlock(wilock)
                    continue
                end
                item = workitems[widx]
                widx = widx + 1
                unlock(wilock)
                times.sched_ovh = times.sched_ovh + toq()
                #ntputs(nodeid, tid, "processing source $item")

                result = InferResult(0, "", 0.0, 0.0, [0.0], 1.0)
                tries = 1 
                while tries <= 3
                    result = try 
                        optimize_source(item, images, catalog, catalog_offset,
                                        rcf_to_index, cache, cache_lock, g_lock,
                                        stagedir, times)
                    catch exc 
                        ntputs(nodeid, tid, "$exc running task $item on try $tries")
                        tries = tries + 1 
                        continue
                    end 
                    break
                end 
                if tries > 3
                    ntputs(nodeid, tid, "exception running task $item on 3 tries, giving up")
                    continue
                end

                lock(results_lock)
                push!(results, result)
                unlock(results_lock)
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

    for tt in ttimes
        add_timing!(timing, tt)
    end

    return results
end


function set_thread_affinity(nid::Int, ppn::Int, tid::Int, nthreads::Int, show_cpu::Bool)
    cpu = (((nid - 1) % ppn) * nthreads)
    show_cpu && ntputs(nid, tid, "bound to $(cpu + tid)")
    mask = zeros(UInt8, 4096)
    mask[cpu + tid] = 1
    uvtid = ccall(:uv_thread_self, UInt64, ())
    ccall(:uv_thread_setaffinity, Int, (Ptr{Void}, Ptr{Void}, Ptr{Void}, Int64),
          pointer_from_objref(uvtid), mask, C_NULL, 4096)
end


function affinitize()
    ppn = try
        parse(Int, ENV["JULIA_EXCLUSIVE"])
    catch exc
        return
    end
    show_cpu = try
        parse(Bool, ENV["CELESTE_SHOW_AFFINITY"])
    catch exc
        false
    end
    function threadfun()
        set_thread_affinity(nodeid, ppn, threadid(), nthreads(), show_cpu)
    end
    ccall(:jl_threading_run, Void, (Any,), Core.svec(threadfun))
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
    affinitize()

    # read the run-camcol-field triplets for this box
    rcfs = get_overlapping_fields(box, stagedir)

    # loads up to 25TB from disk for SDSS
    tic()
    images, catalog_offset, task_offset = load_images(box, rcfs, stagedir)
    timing.load_img = toq()

    try
        t = ENV["CELESTE_EXIT_AFTER_LOAD_IMAGES"]
        exit()
    catch exc
    end

    # loads up to 4TB from disk for SDSS
    tic()
    catalog, tasks = load_catalog(box, rcfs, catalog_offset, task_offset, stagedir)
    timing.load_cat = toq()

    try
        t = ENV["CELESTE_EXIT_AFTER_LOAD_CATALOG"]
        exit()
    catch exc
    end

    if length(tasks) > 0
        # create map from run, camcol, field to index into RCF array
        rcf_to_index = invert_rcf_array(rcfs)

        # optimization -- little disk access, cpu intensive
        results = optimize_sources(images, catalog, tasks,
                                   catalog_offset, task_offset,
                                   rcf_to_index, stagedir, timing)
        timing.num_srcs = length(results)

        tic()
        save_results(outdir, box, results)
        timing.write_results = toq()
    end

    finalize(tasks)
    finalize(catalog)
    finalize(task_offset)
    finalize(catalog_offset)
    finalize(images)
end

