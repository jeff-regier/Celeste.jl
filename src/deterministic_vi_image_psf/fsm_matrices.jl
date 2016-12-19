import Base.DFT: plan_fft!, plan_ifft!, to1
import Base.DFT.FFTW.cFFTWPlan
import ..Log

typealias GMatrix Matrix{SensitiveFloat{Float64}}
typealias fs0mMatrix Matrix{SensitiveFloat{Float64}}
typealias fs1mMatrix Matrix{SensitiveFloat{Float64}}


const plan_fft_lock = Base.Threads.SpinLock()

function __init__()
    global fft_plans, ifft_plans
    fft_plans = Dict{Tuple{Int64, Int64}, cFFTWPlan{Complex{Float64},-1,true,2}}()
    ifft_type = Base.DFT.ScaledPlan{Complex{Float64},Base.DFT.FFTW.cFFTWPlan{Complex{Float64},1,true,2},Float64}
    ifft_plans = Dict{Tuple{Int64, Int64}, ifft_type}()
end


function safe_fft!(A)
    A1 = to1(A)
    global fft_plans

    if !haskey(fft_plans, size(A))
        lock(plan_fft_lock)
        fft_plans[size(A)] = Base.DFT.plan_fft!(A1)
        unlock(plan_fft_lock)
    end

    fft_plans[size(A)] * A1  # mutates A1
    A1
end


function safe_ifft!(A)
    A1 = to1(A)
    global ifft_plans

    if !haskey(ifft_plans, size(A))
        lock(plan_fft_lock)
        ifft_plans[size(A)] = Base.DFT.plan_ifft!(A1)
        unlock(plan_fft_lock)
    end

    ifft_plans[size(A)] * A1  # mutates A1
    A1
end


type FSMSensitiveFloatMatrices
    # The lower corner of the image (in terms of index values)
    h_lower::Int
    w_lower::Int

    fs1m_image::fs1mMatrix;
    fs1m_image_padded::fs1mMatrix;
    fs1m_conv::fs1mMatrix;
    fs1m_conv_padded::fs1mMatrix;

    # We convolve the star fs0m directly using Lanczos interpolation.
    fs0m_conv::fs0mMatrix;

    E_G::GMatrix
    var_G::GMatrix

    # The PSF and its FFT.
    psf_fft::Matrix{Complex{Float64}}
    psf::Matrix{Float64}

    # The amount of padding introduced by the convolution on one side of the
    # image (the total pixels added in each dimension are twice this)
    pad_pix_h::Int
    pad_pix_w::Int

    # Functions for star convolution.
    # kernel_fun is a kernel function that returns the value, derivative, and second
    # derivative of a univariate kernel, e.g. bspline_kernel_with_derivatives()
    kernel_width::Int64
    kernel_fun

    FSMSensitiveFloatMatrices() = begin
        new(1, 1,
            fs1mMatrix(), fs1mMatrix(), fs1mMatrix(), fs1mMatrix(), fs0mMatrix(),
            GMatrix(), GMatrix(),
            Matrix{Complex{Float64}}(),
            Matrix{Float64}(),
            0, 0,
            2, x -> cubic_kernel_with_derivatives(x, 0.0))
    end
end


function initialize_fsm_sf_matrices_band!(
    fsms::FSMSensitiveFloatMatrices,
    s::Int, b::Int, num_active_sources::Int,
    h_lower::Int, w_lower::Int,
    h_upper::Int, w_upper::Int,
    psf_image::Matrix{Float64})

    psf_size = size(psf_image)

    fsms.h_lower = h_lower
    fsms.w_lower = w_lower

    h_width = h_upper - h_lower + 1
    w_width = w_upper - w_lower + 1

    # An fsm value is only sensitive to one source's parameters.
    fsms.fs1m_image = zero_sensitive_float_array(
        Float64, length(GalaxyPosParams), 1, h_width, w_width);
    fsms.fs1m_conv = zero_sensitive_float_array(
        Float64, length(GalaxyPosParams), 1, h_width, w_width);

    fsms.fs0m_conv = zero_sensitive_float_array(
        Float64, length(StarPosParams), 1, h_width, w_width);

    # The amount of padding introduced by the convolution
    (fft_size1, fft_size2) =
        (h_width + psf_size[1] - 1, w_width + psf_size[2] - 1)
    # Make sure that the PSF has an odd dimension.
    @assert psf_size[1] % 2 == 1
    @assert psf_size[2] % 2 == 1
    fsms.pad_pix_h = Integer((psf_size[1] - 1) / 2)
    fsms.pad_pix_w = Integer((psf_size[2] - 1) / 2)

    fsms.fs1m_image_padded = zero_sensitive_float_array(
        Float64, length(GalaxyPosParams), 1, fft_size1, fft_size2);
    fsms.fs1m_conv_padded = zero_sensitive_float_array(
        Float64, length(GalaxyPosParams), 1, fft_size1, fft_size2);

    # Brightness images
    fsms.E_G = zero_sensitive_float_array(
        Float64, length(CanonicalParams), num_active_sources, h_width, w_width);
    fsms.var_G = zero_sensitive_float_array(
        Float64, length(CanonicalParams), num_active_sources, h_width, w_width);

    # Store the psf image and its FFT.
    fsms.psf = deepcopy(psf_image)
    fsms.psf_fft = zeros(Complex{Float64}, fft_size1, fft_size2)
    fsms.psf_fft[1:psf_size[1], 1:psf_size[2]] = fsms.psf

    safe_fft!(fsms.psf_fft)
end


function initialize_fsm_sf_matrices!(
    fsm_mat::Matrix{FSMSensitiveFloatMatrices},
    ea::ElboArgs{Float64},
    psf_image_mat::Matrix{Matrix{Float64}})

    num_active_sources = length(ea.active_sources)

    for n in 1:ea.N, s in 1:ea.S
        p = ea.patches[s, n]
        apb = p.active_pixel_bitmap
        active_cols = find([ any(apb[:, col]) for col in 1:size(apb, 2) ])
        active_rows = find([ any(apb[row, :]) for row in 1:size(apb, 1) ])
        h1 = p.bitmap_offset[1] + minimum(active_rows)
        w1 = p.bitmap_offset[2] + minimum(active_cols)
        h2 = h1 + maximum(active_rows) - 1
        w2 = w1 + maximum(active_cols) - 1

        initialize_fsm_sf_matrices_band!(
            fsm_mat[s, n], s, n, num_active_sources,
            h1, w1, h2, w2, psf_image_mat[s, n])
    end
end


# This is useful for debugging and exploration.
# Do not delete until May 2017.
function debug_populate_fsm_mat!(
    ea::ElboArgs,
    fsm_mat::Matrix{FSMSensitiveFloatMatrices})

    sbs = load_source_brightnesses(ea,
        calculate_gradient=ea.elbo_vars.elbo.has_gradient,
        calculate_hessian=ea.elbo_vars.elbo.has_hessian);

    gal_mcs_vec = Array(Array{GalaxyCacheComponent{Float64}, 4}, ea.N);
    for b=1:ea.N
        gal_mcs_vec[b] = load_gal_bvn_mixtures(
                ea.S, ea.patches, ea.vp, ea.active_sources, b,
                calculate_gradient=ea.elbo_vars.elbo.has_gradient,
                calculate_hessian=ea.elbo_vars.elbo.has_hessian);
    end

    for b=1:ea.N
        for s in 1:ea.S
            fsms = fsm_mat[s, b]
            clear_brightness!(fsms)
            populate_star_fsm_image!(
                ea, s, b, fsms.psf, fsms.fs0m_conv,
                fsms.h_lower, fsms.w_lower,
                fsms.kernel_fun, fsms.kernel_width)
            populate_gal_fsm_image!(
                ea, s, b, gal_mcs_vec[b], fsm_mat[s, n])
            accumulate_source_image_brightness!(
                ea, s, b, fsms, sbs[s])
        end
    end
end
