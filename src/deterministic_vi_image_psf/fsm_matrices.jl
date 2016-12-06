
typealias GMatrix Matrix{SensitiveFloat{Float64}}
typealias fs0mMatrix Matrix{SensitiveFloat{Float64}}
typealias fs1mMatrix Matrix{SensitiveFloat{Float64}}


# Some ideas for optimizing FSMSensitiveFloatMatrices:
# - The default size of the PSF matrix is probably bigger than it
#   needs to be right now.  For most PSFs, most pixels are dark.
# - The fs*m_image matrices are big enough to contain all pixels in the
#   pixel masks.  They could instead be big enough to contain only the active
#   pixels.
# - As-is, a single image only has one set of fs*m_* matrices.  This means
#   each must be large enough to contain all the sources.  If you arranged
#   it so that each source had its own fs*m_* matrices, then they could be
#   much smaller.
# - Currently, active_pixels are set by the size of the image post convolution.
#   However, fs1m_image only needs to be as big as the image pre convolution.
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

    # A vector of psfs, one for each source.
    psf_fft_vec::Vector{Matrix{Complex{Float64}}}
    psf_vec::Vector{Matrix{Float64}}

    # The amount of padding introduced by the convolution on one side of the
    # image (the total pixels added in each dimension are twice this)
    pad_pix_h::Int
    pad_pix_w::Int

    FSMSensitiveFloatMatrices() = begin
        new(1, 1,
            fs1mMatrix(), fs1mMatrix(), fs1mMatrix(), fs1mMatrix(), fs0mMatrix(),
            GMatrix(), GMatrix(),
            Vector{Matrix{Complex{Float64}}}(),
            Vector{Matrix{Float64}}(),
            0, 0)
    end
end


function initialize_fsm_sf_matrices_band!(
    fsms::FSMSensitiveFloatMatrices,
    b::Int,
    num_active_sources::Int,
    h_lower::Int, w_lower::Int,
    h_upper::Int, w_upper::Int,
    psf_image_mat::Matrix{Matrix{Float64}})

    # Require that every PSF is the same size, since we are only
    # keeping one padded matrix for the whole band.
    psf_sizes = Set([ size(im) for im in psf_image_mat[:, b] ])
    @assert length(psf_sizes) == 1
    psf_size = pop!(psf_sizes)

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
    S = size(psf_image_mat, 1)
    fsms.psf_fft_vec = Array(Matrix{Complex{Float64}}, S)
    fsms.psf_vec = Array(Matrix{Float64}, S)
    for s in 1:size(psf_image_mat, 1)
        fsms.psf_fft_vec[s] =
            zeros(Complex{Float64}, fft_size1, fft_size2);
        fsms.psf_fft_vec[s][1:psf_size[1], 1:psf_size[2]] =
            psf_image_mat[s, b];
        fft!(fsms.psf_fft_vec[s]);
        fsms.psf_vec[s] = psf_image_mat[s, b]
    end
end


function initialize_fsm_sf_matrices!(
    fsm_vec::Vector{FSMSensitiveFloatMatrices},
    ea::ElboArgs{Float64},
    psf_image_mat::Matrix{Matrix{Float64}})

    # Get the extreme active pixels in each band.
    h_lower_vec = Int[typemax(Int) for b in ea.images ]
    w_lower_vec = Int[typemax(Int) for b in ea.images ]
    h_upper_vec = Int[0 for b in ea.images ]
    w_upper_vec = Int[0 for b in ea.images ]

    # Since we are using the same size fsm matrix for each source,
    # and even non-active sources need to be rendered, we need to make
    #.the fsm matrices as big as the total number of active pixels.
    # Right now I just set it as big as the entire active pixel bitmap.
    for s in 1:ea.S, n in 1:ea.N
        p = ea.patches[s, n]
        h1 = p.bitmap_offset[1] + 1
        w1 = p.bitmap_offset[2] + 1
        h2 = h1 + size(p.active_pixel_bitmap, 1) - 1
        w2 = w1 + size(p.active_pixel_bitmap, 2) - 1
        h_lower_vec[n] = min(h_lower_vec[n], h1)
        h_upper_vec[n] = max(h_upper_vec[n], h2)
        w_lower_vec[n] = min(w_lower_vec[n], w1)
        w_upper_vec[n] = max(w_upper_vec[n], w2)
    end

    num_active_sources = length(ea.active_sources)

    # Pre-allocate arrays.
    for b in 1:ea.N
        initialize_fsm_sf_matrices_band!(
            fsm_vec[b], b, num_active_sources,
            h_lower_vec[b], w_lower_vec[b],
            h_upper_vec[b], w_upper_vec[b],
            psf_image_mat)
    end
end


# This is useful for debugging and exploration.
# Do not delete until May 2017.
function debug_populate_fsm_vec!(
    ea::ElboArgs,
    fsm_vec::Array{FSMSensitiveFloatMatrices},
    lanczos_width::Int)

    sbs = load_source_brightnesses(ea,
        calculate_gradient=ea.elbo_vars.calculate_gradient,
        calculate_hessian=ea.elbo_vars.calculate_hessian);

    gal_mcs_vec = Array(Array{GalaxyCacheComponent{Float64}, 4}, ea.N);
    for b=1:ea.N
        gal_mcs_vec[b] = load_gal_bvn_mixtures(
                ea.S, ea.patches, ea.vp, ea.active_sources, b,
                calculate_gradient=ea.elbo_vars.calculate_gradient,
                calculate_hessian=ea.elbo_vars.calculate_hessian);
    end

    for b=1:ea.N
        for s in 1:ea.S
            populate_star_fsm_image!(
                ea, s, b, fsm_vec[b].psf_vec[s], fsm_vec[b].fs0m_conv,
                fsm_vec[b].h_lower, fsm_vec[b].w_lower, lanczos_width)
            populate_gal_fsm_image!(
                ea, s, b, gal_mcs_vec[b], fsm_vec[b])
            accumulate_source_image_brightness!(
                ea, s, b, fsm_vec[b], sbs[s])
        end
    end
end
