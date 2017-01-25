"""
Calculate value, gradient, and hessian of the variational ELBO using a PSF
image rather than a mixture of Gaussians.
"""

module DeterministicVIImagePSF

using StaticArrays, DiffBase

import ..DeterministicVI:
    ElboArgs, ElboIntermediateVariables, maximize_f,
    StarPosParams, GalaxyPosParams, CanonicalParams, VariationalParams,
    SourceBrightness, GalaxyComponent, SkyPatch,
    load_source_brightnesses, add_elbo_log_term!,
    accumulate_source_pixel_brightness!,
    KLDivergence, init_sources

import ..Model:
    populate_gal_fsm!, getids, ParamSet, linear_world_to_pix, lidx,
    BvnComponent, GalaxyCacheComponent, GalaxySigmaDerivs,
    get_bvn_cov, galaxy_prototypes, linear_world_to_pix,
    Image, eval_psf, CatalogEntry

import ..SensitiveFloats:
    SensitiveFloat, zero_sensitive_float_array,
    multiply_sfs!, add_scaled_sfs!, clear!

import ..Infer: load_active_pixels!, get_sky_patches

import ..PSF: get_psf_at_point, trim_psf

import WCS

include("deterministic_vi_image_psf/sensitive_float_fft.jl")
include("deterministic_vi_image_psf/kernels.jl")
include("deterministic_vi_image_psf/fsm_matrices.jl")
include("deterministic_vi_image_psf/elbo_image_psf.jl")

export elbo_likelihood_with_fft!, FSMSensitiveFloatMatrices,
       initialize_fsm_sf_matrices!, initialize_fft_elbo_parameters,
       get_fft_elbo_function, load_fsm_mat


function infer_source_fft(images::Vector{Image},
                          neighbors::Vector{CatalogEntry},
                          entry::CatalogEntry)
   cat_local = vcat([entry], neighbors)
   vp = init_sources([1], cat_local)
   patches = get_sky_patches(images, cat_local)
   load_active_pixels!(images, patches)

   ea_fft, fsm_mat = initialize_fft_elbo_parameters(
       images, vp, patches, [1], use_raw_psf=false)
   elbo_fft_opt = get_fft_elbo_function(ea_fft, fsm_mat)
   maximize_f(elbo_fft_opt, ea_fft, max_iters=150)

   vp[1]
end

end
