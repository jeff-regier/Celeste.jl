"""
Self-contained wrapper of C library based on Source Extractor
(https://github.com/kbarbary/sep) for performing image background estimation
and segmentation (detection). This is used in initialization in Celeste.

In the future, this module could be replaced by a pure-Julia implementation.
The background could be made an explicit part of the model.
"""

module SEP

import Base: collect, broadcast!, broadcast, -, show

if isfile(joinpath(dirname(@__FILE__),"..","deps","deps.jl"))
    include("../deps/deps.jl")
else
    error("Celeste.SEP not properly installed. Please run Pkg.build(\"Celeste\")")
end

# Julia Type -> numeric type code from sep.h
const SEPMaskType = Union{Bool, UInt8, Cint, Cfloat, Cdouble}
const PixType = Union{UInt8, Cint, Cfloat, Cdouble}
sep_typecode(::Type{Bool}) = Cint(11)
sep_typecode(::Type{UInt8}) = Cint(11)
sep_typecode(::Type{Cint}) = Cint(31)
sep_typecode(::Type{Cfloat}) = Cint(42)
sep_typecode(::Type{Cdouble}) = Cint(82)


# definitions from sep.h
const SEP_NOISE_NONE = Cshort(0)
const SEP_NOISE_STDDEV = Cshort(1)
const SEP_NOISE_VAR = Cshort(2)
const SEP_THRESH_REL = Cint(0)
const SEP_THRESH_ABS = Cint(1)
const SEP_FILTER_CONV = Cint(0)
const SEP_FILTER_MATCHED = Cint(1)

function sep_assert_ok(status::Cint)
    if status != 0
        msg = Vector{UInt8}(61)
        ccall((:sep_get_errmsg, libsep), Void, (Int32,Ptr{UInt8}), status, msg)
        msg[end] = 0  # ensure NULL-termination, just in case.
        error(unsafe_string(pointer(msg)))
    end
end

# internal use only: mirrors `sep_image` struct in sep.h
struct sep_image
data::Ptr{Void}
noise::Ptr{Void}
mask::Ptr{Void}
dtype::Cint
ndtype::Cint
mdtype::Cint
w::Cint
h::Cint
noiseval::Cdouble
noise_type::Cshort
gain::Cdouble
maskthresh::Cdouble
end

function sep_image(data::Array{T, 2};
                   noise=nothing, mask=nothing, noise_type=:stddev,
                   gain=0.0, mask_thresh=0.0) where {T<:PixType}
    sz = size(data)

    # data is required
    data_ptr = Ptr{Void}(pointer(data))
    dtype = sep_typecode(T)

    # mask options
    mask_ptr = C_NULL
    mdtype = Cint(0)
    if mask !== nothing
        isa(mask, Matrix) || error("mask must be a 2-d array")
        size(mask) == sz || error("data and mask must be same size")
        mask_ptr = Ptr{Void}(pointer(mask))
        mdtype = sep_typecode(eltype(mask))
    end

    # noise options
    ndtype = Cint(0)
    noise_ptr = C_NULL
    noiseval = 0.0
    noise_typecode = SEP_NOISE_NONE
    if noise !== nothing
        if isa(noise, Matrix)
            size(noise) == sz || error("noise array must be same size as data")
            noise_ptr = Ptr{Void}(pointer(noise))
            ndtype = sep_typecode(eltype(noise))
        elseif isa(noise, Real)
            noiseval = Cdouble(noise)
        else
            error("noise must be array or number")
        end
        noise_typecode = ((noise_type == :stddev) ? SEP_NOISE_STDDEV :
                          (noise_type == :var) ? SEP_NOISE_VAR :
                          error("noise_type must be :stddev or :var"))
    end

    return sep_image(data_ptr, noise_ptr, mask_ptr,
                     dtype, ndtype, mdtype,
                     sz[1], sz[2],
                     noiseval, noise_typecode,
                     Cdouble(gain),
                     Cdouble(mask_thresh))
end

# ---------------------------------------------------------------------------
# Background functions

"""
    Background(data::Matrix; <keyword arguments>)

Spline representation of the variable background and noise of an image.

# Arguments

- `data::Matrix`:
- `mask=nothing`:
- `boxsize=(64, 64)`:
- `filtersize=(3, 3)`:
- `filterthresh=0.0`:
- `mask_thresh=0`:
"""
mutable struct Background
    ptr::Ptr{Void}
    data_size::Tuple{Int, Int}

    function Background(data::Array{T, 2} where T<:PixType;
                        mask=nothing, boxsize=(64, 64), filtersize=(3, 3),
                        filterthresh=0.0, mask_thresh=0)
        im = sep_image(data; mask=mask, mask_thresh=mask_thresh)
        result = Ref{Ptr{Void}}(C_NULL)
        status = ccall((:sep_background, libsep), Cint,
                       (Ptr{sep_image}, Cint, Cint, Cint, Cint, Cdouble,
                        Ref{Ptr{Void}}),
                       &im, boxsize[1], boxsize[2], filtersize[1],
                       filtersize[2], filterthresh, result)
        sep_assert_ok(status)
        bkg = new(result[], size(data))
        finalizer(bkg, free!)
        return bkg
    end
end

free!(bkg::Background) = ccall((:sep_bkg_free, libsep), Void, (Ptr{Void},),
                               bkg.ptr)

global_mean(bkg::Background) = ccall((:sep_bkg_global, libsep), Cfloat,
                                     (Ptr{Void},), bkg.ptr)

global_rms(bkg::Background) = ccall((:sep_bkg_globalrms, libsep), Cfloat,
                                    (Ptr{Void},), bkg.ptr)

function show(io::IO, bkg::Background)
    print(io, "Background $(bkg.data_size[1])×$(bkg.data_size[2])\n")
    print(io, " - global mean: $(global_mean(bkg))\n")
    print(io, " - global rms : $(global_rms(bkg))\n")
end

function collect(::Type{T}, bkg::Background) where {T<:PixType}
    result = Array{T}(bkg.data_size)
    status = ccall((:sep_bkg_array, libsep), Cint, (Ptr{Void}, Ptr{Void}, Cint),
                   bkg.ptr, result, sep_typecode(T))
    sep_assert_ok(status)
    return result
end
# default collection type is Float32, because that's what's natively stored in
# background.
# TODO: make default the input array type in `background` instead?
collect(bkg::Background) = collect(Float32, bkg)


"""
    rms(bkg)
    rms(T, bkg)

Return an array of the standard deviation of the background. The result is
the size of the original image and is of type `T`, if given.
"""
function rms(::Type{T}, bkg::Background) where {T<:PixType}
    result = Array{T}(bkg.data_size)
    status = ccall((:sep_bkg_rmsarray, libsep), Cint,
                   (Ptr{Void}, Ptr{Void}, Cint),
                   bkg.ptr, result, sep_typecode(T))
    sep_assert_ok(status)
    return result
end
rms(bkg::Background) = rms(Float32, bkg)


# In-place background subtraction: A .-= bkg
function broadcast!(-, A::Array{T, 2},  ::Array{T, 2}, bkg::Background) where {T<:PixType}
    if size(A) != bkg.data_size
         throw(DimensionMismatch("dimensions must match"))
    end
    status = ccall((:sep_bkg_subarray, libsep), Cint,
                   (Ptr{Void}, Ptr{Void}, Cint),
                   bkg.ptr, A, sep_typecode(T))
    sep_assert_ok(status)
end
function broadcast(-, A::Array{T, 2} where T<:PixType, bkg::Background)
    B = copy(A)
    B .-= bkg
    return B
end

function (-)(A::Array{T, 2}, bkg::Background) where {T<:PixType}
    B = copy(A)
    B .-= bkg
    return B
end


# -----------------------------------------------------------------------------
# Source Extraction

# Internal use only: Mirror of C struct
struct sep_catalog
    nobj::Cint                 # number of objects (length of all arrays)
    thresh::Ptr{Cfloat}              # threshold (ADU)
    npix::Ptr{Cint}              # # pixels extracted (size of pix array)
    tnpix::Ptr{Cint}                # # pixels above thresh (unconvolved)
    xmin::Ptr{Cint}
    xmax::Ptr{Cint}
    ymin::Ptr{Cint}
    ymax::Ptr{Cint}
    x::Ptr{Cdouble}              # barycenter (first moments)
    y::Ptr{Cdouble}
    x2::Ptr{Cdouble}     # second moments
    y2::Ptr{Cdouble}
    xy::Ptr{Cdouble}
    errx2::Ptr{Cdouble}  # second moment errors
    erry2::Ptr{Cdouble}
    errxy::Ptr{Cdouble}
    a::Ptr{Cfloat}       # ellipse parameters
    b::Ptr{Cfloat}
    theta::Ptr{Cfloat}
    cxx::Ptr{Cfloat}     # alternative ellipse parameters
    cyy::Ptr{Cfloat}
    cxy::Ptr{Cfloat}
    cflux::Ptr{Cfloat}   # total flux of pixels (convolved)
    flux::Ptr{Cfloat}    # total flux of pixels (unconvolved)
    cpeak::Ptr{Cfloat}   # peak pixel flux (convolved)
    peak::Ptr{Cfloat}    # peak pixel flux (unconvolved)
    xcpeak::Ptr{Cint}    # x, y coords of peak (convolved) pixel
    ycpeak::Ptr{Cint}
    xpeak::Ptr{Cint}     # x, y coords of peak (unconvolved) pixel
    ypeak::Ptr{Cint}
    flag::Ptr{Cshort}    # extraction flags
    pix::Ptr{Ptr{Cint}}  # array giving indicies of object's pixels in
                         # image (linearly indexed). Length is `npix`.
                         # (pointer to within the `objectspix` buffer)
    objectspix::Ptr{Cint}  # buffer holding pixel indicies for all objects
end


struct Catalog
npix::Vector{Cint}
xmin::Vector{Cint}
xmax::Vector{Cint}
ymin::Vector{Cint}
ymax::Vector{Cint}
x::Vector{Cdouble}
y::Vector{Cdouble}
a::Vector{Cfloat}
b::Vector{Cfloat}
theta::Vector{Cfloat}
cxx::Vector{Cfloat}
cyy::Vector{Cfloat}
cxy::Vector{Cfloat}
flux::Vector{Cfloat}
pix::Vector{Vector{Cint}}
end

function show(io::IO, cat::Catalog)
    print(io, "SEP.Catalog with $(length(cat.x)) entries")
end

"""
    unsafe_copy(src::Ptr{T}, N)

Create a `Vector{T}` and copy `N` elements from `Ptr{T}` into it.
"""
function unsafe_copy(src::Ptr{T}, N) where {T}
    dest = Vector{T}(N)
    unsafe_copy!(pointer(dest), src, N)
    return dest
end

"""
    extract(data, thresh; <keyword arguments>)

Perform image segmentation on the data and return a catalog of sources.

# Arguments

- `data::Matrix`: Image data.
- `thresh::Real`: Threshold for detection in absolute units, or in standard
  deviations if noise is given.
- `noise=nothing`:
- `noise_type=:stddev`:
- `mask=nothing`:
- `mask_thresh=0`:
- `minarea=5`:
- `filter_kernel`:
- `filter_type=:conv`:
- `deblend_nthresh=32`:
- `deblend_cont=0.005`: Minimum contrast in deblending. Set to 1.0 to disable
  deblending.
- `clean=true`: Perform cleaning?
- `clean_param=1.0`
- `gain=0.0`
"""
function extract(data::Array{T, 2} where T, thresh::Real;
                 noise=nothing, mask=nothing, noise_type=:stddev,
                 minarea=5,
                 filter_kernel=Float32[1 2 1; 2 4 2; 1 2 1],
                 filter_type=:convolution,
                 deblend_nthresh=32, deblend_cont=0.005,
                 clean=true, clean_param=1.0,
                 gain=0.0, mask_thresh=0)

    im = sep_image(data; noise=noise, mask=mask, noise_type=noise_type,
                   gain=gain, mask_thresh=mask_thresh)

    thresh_typecode = noise === nothing ? SEP_THRESH_ABS : SEP_THRESH_REL

    filter_typecode = ((filter_type == :matched)? SEP_FILTER_MATCHED :
                       (filter_type == :convolution)? SEP_FILTER_CONV :
                       error("filter_type must be :matched or :convolution"))

    # convert filter kernel to Cfloat array
    filter_kernel_cfloat = convert(Array{Cfloat, 2}, filter_kernel)
    filter_size = size(filter_kernel_cfloat)

    ccatalog_ptr_ref = Ref{Ptr{sep_catalog}}(C_NULL)
    status = ccall((:sep_extract, libsep), Cint,
                   (Ptr{sep_image},
                    Cfloat, Cint, # thresh, thresh_type
                    Cint,  # minarea
                    Ptr{Cfloat}, Cint, Cint,  # conv, convw, convh
                    Cint, # filter_type
                    Cint, Cdouble,  # deblend_nthresh, deblend_cont
                    Cint, Cdouble,  # clean_flag, clean_param
                    Ref{Ptr{sep_catalog}}),
                   &im,
                   thresh, thresh_typecode,
                   minarea,
                   filter_kernel_cfloat, filter_size[1], filter_size[2],
                   filter_typecode,
                   deblend_nthresh, deblend_cont,
                   clean, clean_param,
                   ccatalog_ptr_ref)
    sep_assert_ok(status)

    # get pointer to C-allocated result
    ccatalog_ptr = ccatalog_ptr_ref[]

    # copy result arrays into jula-managed memory
    ccatalog = unsafe_load(ccatalog_ptr)
    nobj = ccatalog.nobj
    # translate pixel index vectors
    npix = unsafe_copy(ccatalog.npix, nobj)
    pix = Vector{Vector{Cint}}(nobj)
    for i in eachindex(pix)
        ptr = unsafe_load(ccatalog.pix, i)  # ptr to pixel indicies for i-th obj
        pix[i] = unsafe_copy(ptr, npix[i])
        pix[i] .+= Cint(1) # change to 1-indexing
    end

    result = Catalog(npix,
                     unsafe_copy(ccatalog.xmin, nobj),
                     unsafe_copy(ccatalog.xmax, nobj),
                     unsafe_copy(ccatalog.ymin, nobj),
                     unsafe_copy(ccatalog.ymax, nobj),
                     unsafe_copy(ccatalog.x, nobj),
                     unsafe_copy(ccatalog.y, nobj),
                     unsafe_copy(ccatalog.a, nobj),
                     unsafe_copy(ccatalog.b, nobj),
                     unsafe_copy(ccatalog.theta, nobj),
                     unsafe_copy(ccatalog.cxx, nobj),
                     unsafe_copy(ccatalog.cyy, nobj),
                     unsafe_copy(ccatalog.cxy, nobj),
                     unsafe_copy(ccatalog.flux, nobj),
                     pix)

    # switch to 1 indexing
    result.x .+= 1.0
    result.y .+= 1.0

    # free result allocated in sep_extract
    ccall((:sep_catalog_free, libsep), Void, (Ptr{sep_catalog},), ccatalog_ptr)

    return result
end


# mask ellipse
"""
    mask_ellipse(data, x, y, cxx, cyy, cxy; r=1.0)

Set values within ellipse to `true`.
"""
function mask_ellipse(data::Array{T, 2} where T <: Union{UInt8, Bool},
                      x::Real, y::Real, cxx::Real, cyy::Real, cxy::Real;
                      r=1.0)
    w, h = size(data)
    ccall((:sep_set_ellipse, libsep), Void,
          (Ptr{Cuchar}, Cint, Cint, Cdouble, Cdouble, Cdouble, Cdouble,
           Cdouble, Cdouble, Cuchar), data, w, h, x-1.0, y-1.0, cxx, cyy,
          cxy, r, true)
end

end # module
