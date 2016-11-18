using Celeste: Model, DeterministicVI
using StaticArrays

import Celeste: Infer, DeterministicVI, ParallelRun
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

include("celeste_tools/celeste_tools.jl")
const dir = "/home/rgiordan/Documents/git_repos/CelesteDev.jl/"

# include(joinpath(dir, "rasterized_psf/lanczos.jl"))
include(joinpath(dir, "rasterized_psf/elbo_pixelated_psf.jl"))

using DeterministicVIImagePSF
import Synthetic
using SampleData

using Base.Test
using Distributions

using DeterministicVI.CanonicalParams
using SensitiveFloats.zero_sensitive_float
using SensitiveFloats.zero_sensitive_float_array
using SensitiveFloats.SensitiveFloat
using SensitiveFloats.clear!


using ForwardDiff
using DeterministicVIImagePSF

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
    psf_image_mat = Matrix{Matrix{Float64}}(1, 1);
    psf_image = zeros(3, 3);
    psf_image[2, 2] = 0.5;
    psf_image[2, 1] = psf_image[1, 2] = 0.25;
    psf_image_mat[1, 1] = psf_image;
    DeterministicVIImagePSF.initialize_fsm_sf_matrices_band!(
        fsms, 1, 1, 1, 1, 3, 3, psf_image_mat)

    sf = zero_sensitive_float(GalaxyPosParams, Float64)
    sf.v[1] = 3;
    sf.d[:, 1] = rand(size(sf.d, 1))
    h = rand(size(sf.h))
    sf.h = h * h';
    fsms.fs1m_image_padded[2, 2] = deepcopy(sf);
    DeterministicVIImagePSF.convolve_sensitive_float_matrix!(
        fsms.fs1m_image_padded, fsms.psf_fft_vec[1], fsms.fs1m_conv_padded);
    h_indices = (1:3) + fsms.pad_pix_h
    w_indices = (1:3) + fsms.pad_pix_w
    conv_image =
        Float64[ sf.v[1] for sf in fsms.fs1m_conv_padded ][h_indices, w_indices];
    @test_approx_eq(sf.v[1] * psf_image, conv_image)

    for ind in 1:size(sf.d, 1)
        conv_image =
            Float64[ sf.d[ind] for sf in fsms.fs1m_conv_padded ][h_indices, w_indices];
        @test_approx_eq(sf.d[ind] * psf_image, conv_image)
    end

    for ind1 in 1:size(sf.h, 1), ind2 in 1:size(sf.h, 2)
        conv_image =
            Float64[ sf.h[ind1, ind2] for sf in fsms.fs1m_conv_padded ][h_indices, w_indices];
        @test_approx_eq(sf.h[ind1, ind2] * psf_image, conv_image)
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

    @test_approx_eq sinc(x) v
    @test_approx_eq fd_v v
    @test_approx_eq fd_d d
    @test_approx_eq fd_h h
end


function test_lanczos_kernel()
    psf_image = zeros(Float64, 5, 5);
    psf_image[3, 3] = 0.5
    psf_image[2, 3] = psf_image[3, 2] = psf_image[4, 3] = psf_image[3, 4] = 0.125

    star_loc = Float64[5.1, 5.2]
    lanczos_width = 2.0

    function lanczos_kernel_fd{NumType <: Number}(x_vec::Vector{NumType})
        v, d, h = DeterministicVIImagePSF.lanczos_kernel_with_derivatives_nocheck(
            x_vec[1], lanczos_width)
        return v
    end

    x = 0.7
    fd_v = lanczos_kernel_fd([ x ])
    fd_d = ForwardDiff.gradient(lanczos_kernel_fd, Float64[ x ])[1]
    fd_h = ForwardDiff.hessian(lanczos_kernel_fd, Float64[ x ])[1, 1]

    v, d, h = DeterministicVIImagePSF.lanczos_kernel_with_derivatives_nocheck(x, lanczos_width)

    @test_approx_eq fd_v v
    @test_approx_eq fd_d d
    @test_approx_eq fd_h h
end


function test_lanczos_interpolate()
    psf_image = zeros(Float64, 5, 5);
    psf_image[3, 3] = 0.5
    psf_image[2, 3] = psf_image[3, 2] = psf_image[4, 3] = psf_image[3, 4] = 0.125

    local wcs_jacobian = Float64[0.9 0.2; 0.1 0.8]
    local world_loc = Float64[5.1, 5.2]
    local lanczos_width = 2

    image_size = (11, 11)
    function lanczos_interpolate_loc{T <: Number}(
        world_loc::Vector{T}, calculate_derivs::Bool)
        local image = zero_sensitive_float_array(StarPosParams, T, 1, image_size...);
        local pixel_loc = Celeste.Model.linear_world_to_pix(
            wcs_jacobian, Float64[0., 0.], Float64[1.0, 0.5], world_loc)
        DeterministicVIImagePSF.lanczos_interpolate!(
            image, psf_image, pixel_loc, lanczos_width, wcs_jacobian,
            calculate_derivs, calculate_derivs)
        return image
    end

    image = lanczos_interpolate_loc(world_loc, true)
    for test_pix in prod(image_size)
        function lanczos_interpolate_loc_fd{T <: Number}(world_loc::Vector{T})
            local image = lanczos_interpolate_loc(world_loc, false)
            return image[test_pix].v[1]
        end

        fd_v = lanczos_interpolate_loc_fd(world_loc)
        fd_d = ForwardDiff.gradient(lanczos_interpolate_loc_fd, world_loc)
        fd_h = ForwardDiff.hessian(lanczos_interpolate_loc_fd, world_loc)

        @test_approx_eq image[test_pix].v[1] fd_v
        @test_approx_eq image[test_pix].d fd_d
        @test_approx_eq image[test_pix].h fd_h
    end
end



######################
test_sinc()
test_lanczos_kernel()
test_convolve_sensitive_float_matrix()
test_lanczos_interpolate()
