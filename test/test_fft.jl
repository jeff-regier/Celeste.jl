using Celeste.DeterministicVIImagePSF

using Celeste.DeterministicVI.CanonicalParams
using Celeste.SensitiveFloats.zero_sensitive_float_array
using Celeste.SensitiveFloats.SensitiveFloat
using Celeste.SensitiveFloats.clear!

using ForwardDiff
using Base.Test


# Overload some base functions for testing with ForwardDiff
import Base.floor
function floor{N}(x::ForwardDiff.Dual{N,Float64})
    return floor(x.value)
end

function floor{N}(x::ForwardDiff.Dual{N,ForwardDiff.Dual{N,Float64}})
    return floor(x.value)
end


import Base.sinc
function sinc{N}(x::ForwardDiff.Dual{N,Float64})
    # Note that this won't work with x = 0, but it's ok for testing.
    return sin(x * pi) / (x * pi)
end

function sinc{N}(x::ForwardDiff.Dual{N,ForwardDiff.Dual{N,Float64}})
    # Note that this won't work with x = 0.
    return sin(x * pi) / (x * pi)
end


function test_convolve_sensitive_float_matrix()
    # Use the FSMSensitiveFloatMatrices because it initializes all the
    # sizes for us automatically.
    fsms = DeterministicVIImagePSF.FSMSensitiveFloatMatrices();
    psf_image = zeros(3, 3);
    psf_image[2, 2] = 0.5;
    psf_image[2, 1] = psf_image[1, 2] = 0.25;
    DeterministicVIImagePSF.initialize_fsm_sf_matrices_band!(
        fsms, 1, 1,
        1, 1, 3, 3, psf_image)

    sf = SensitiveFloat{Float64}(length(GalaxyPosParams), 1, true, true)
    sf.v[] = 3;
    sf.d[:, 1] = rand(size(sf.d, 1))
    h = rand(size(sf.h))
    sf.h[:] = h * h';
    fsms.fs1m_image_padded[2, 2] = deepcopy(sf);
    DeterministicVIImagePSF.convolve_sensitive_float_matrix!(
        fsms.fs1m_image_padded, fsms.psf_fft, fsms.fs1m_conv_padded);
    h_indices = (1:3) + fsms.pad_pix_h
    w_indices = (1:3) + fsms.pad_pix_w
    conv_image =
        Float64[ sf.v[] for sf in fsms.fs1m_conv_padded ][h_indices, w_indices];
    @test sf.v[] * psf_image ≈ conv_image

    for ind in 1:size(sf.d, 1)
        conv_image =
            Float64[ sf.d[ind] for sf in fsms.fs1m_conv_padded ][h_indices, w_indices];
        @test sf.d[ind] * psf_image ≈ conv_image
    end

    for ind1 in 1:size(sf.h, 1), ind2 in 1:size(sf.h, 2)
        conv_image =
            Float64[ sf.h[ind1, ind2] for sf in fsms.fs1m_conv_padded ][h_indices, w_indices];
        @test sf.h[ind1, ind2] * psf_image ≈ conv_image
    end


end


function test_sinc()
    function sinc_with_derivatives_fd{T <: Number}(x::Vector{T})
        v, d, h = DeterministicVIImagePSF.sinc_with_derivatives(x[1])
        return v
    end

    x = 0.7
    fd_v = sinc_with_derivatives_fd(Float64[ x ])
    fd_d = ForwardDiff.gradient(sinc_with_derivatives_fd, Float64[ x ])[1]
    fd_h = ForwardDiff.hessian(sinc_with_derivatives_fd, Float64[ x ])[1, 1]

    v, d, h = DeterministicVIImagePSF.sinc_with_derivatives(x)

    @test sinc(x) ≈ v
    @test fd_v    ≈ v
    @test fd_d    ≈ d
    @test fd_h    ≈ h
end


function test_lanczos_kernel()
    kernel_width = 2.0
    function lanczos_kernel_fd{NumType <: Number}(x_vec::Vector{NumType})
        v, d, h = DeterministicVIImagePSF.lanczos_kernel_with_derivatives_nocheck(
            x_vec[1], kernel_width)
        return v
    end

    for x in Float64[-2.2, -1.2, -0.2, 0, 0.2, 1.2, 2.2]
        fd_v = lanczos_kernel_fd([ x ])
        fd_d = ForwardDiff.gradient(lanczos_kernel_fd, Float64[ x ])[1]
        fd_h = ForwardDiff.hessian(lanczos_kernel_fd, Float64[ x ])[1, 1]

        v, d, h = DeterministicVIImagePSF.lanczos_kernel_with_derivatives_nocheck(x, kernel_width)

        @test fd_v ≈ v
        if x == 0
            @test_broken fd_d ≈ d
        else
            @test fd_h ≈ h
        end
    end
end


function test_bspline_kernel()
    function bspline_kernel_fd{NumType <: Number}(x_vec::Vector{NumType})
        v, d, h = DeterministicVIImagePSF.bspline_kernel_with_derivatives(x_vec[1])
        return v
    end

    for x in Float64[-2.2, -1.2, -0.2, 0, 0.2, 1.2, 2.2]
        fd_v = bspline_kernel_fd([ x ])
        fd_d = ForwardDiff.gradient(bspline_kernel_fd, Float64[ x ])[1]
        fd_h = ForwardDiff.hessian(bspline_kernel_fd, Float64[ x ])[1, 1]

        v, d, h = DeterministicVIImagePSF.bspline_kernel_with_derivatives(x)

        @test fd_v ≈ v
        @test fd_d ≈ d
        @test fd_h ≈ h
    end
end


function test_cubic_kernel()
    kernel_param = -0.75

    function cubic_kernel_fd{NumType <: Number}(x_vec::Vector{NumType})
        v, d, h = DeterministicVIImagePSF.cubic_kernel_with_derivatives(
            x_vec[1], kernel_param)
        return v
    end

    for x in Float64[-2.2, -1.2, -0.2, 0, 0.2, 1.2, 2.2]
        fd_v = cubic_kernel_fd([ x ])
        fd_d = ForwardDiff.gradient(cubic_kernel_fd, Float64[ x ])[1]
        fd_h = ForwardDiff.hessian(cubic_kernel_fd, Float64[ x ])[1, 1]

        v, d, h = DeterministicVIImagePSF.cubic_kernel_with_derivatives(x, kernel_param)

        @test fd_v ≈ v
        @test fd_d ≈ d
        @test fd_h ≈ h
    end
end


function test_interpolate()
    psf_image = zeros(Float64, 5, 5);
    psf_image[3, 3] = 0.5
    psf_image[2, 3] = psf_image[3, 2] = psf_image[4, 3] = psf_image[3, 4] = 0.125

    local wcs_jacobian = Float64[0.9 0.2; 0.1 0.8]
    local world_loc = Float64[5.1, 5.2]
    local kernel_width = 2

    image_size = (11, 11)

    function test_kernel(kernel_fun)
        function interpolate_loc{T <: Number}(
            world_loc::Vector{T}, calculate_gradient::Bool)
            local image = zero_sensitive_float_array(T, length(StarPosParams), 1,
                                                                image_size...)
            local pixel_loc = Celeste.Model.linear_world_to_pix(
                wcs_jacobian, Float64[0., 0.], Float64[1.0, 0.5], world_loc)
            DeterministicVIImagePSF.interpolate!(
                kernel_fun, kernel_width, image, psf_image, pixel_loc, wcs_jacobian,
                calculate_gradient, calculate_gradient)
            return image
        end

        image = interpolate_loc(world_loc, true)
        for test_pix in prod(image_size)
            function interpolate_loc_fd{T <: Number}(world_loc::Vector{T})
                local image = interpolate_loc(world_loc, false)
                return image[test_pix].v[]
            end

            fd_v = interpolate_loc_fd(world_loc)
            fd_d = ForwardDiff.gradient(interpolate_loc_fd, world_loc)
            fd_h = ForwardDiff.hessian(interpolate_loc_fd, world_loc)

            @test image[test_pix].v[] ≈ fd_v
            @test image[test_pix].d   ≈ fd_d
            @test image[test_pix].h   ≈ fd_h
        end
    end

    println("Testing cubic kernel")
    test_kernel(x ->
        DeterministicVIImagePSF.cubic_kernel_with_derivatives(x, -0.75))

    println("Testing bspline kernel")
    test_kernel(DeterministicVIImagePSF.bspline_kernel_with_derivatives)

    println("Testing lanczos kernel")
    test_kernel(x ->
        DeterministicVIImagePSF.lanczos_kernel_with_derivatives(
            x, Float64(kernel_width)))
end


test_sinc()
test_lanczos_kernel()
test_bspline_kernel()
test_cubic_kernel()
test_convolve_sensitive_float_matrix()
test_interpolate()
