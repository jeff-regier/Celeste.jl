"""
Calculate value, gradient, and hessian of the variational ELBO using a PSF
image rather than a mixture of Gaussians.
"""

module DeterministicVIImagePSF

using StaticArrays

import ..DeterministicVI:
    ElboArgs, ElboIntermediateVariables,
    StarPosParams, GalaxyPosParams, CanonicalParams,
    SourceBrightness, GalaxyComponent, SkyPatch,
    load_source_brightnesses, add_elbo_log_term!,
    accumulate_source_pixel_brightness!

import ..Model:
    populate_gal_fsm!, getids, ParamSet, linear_world_to_pix, lidx,
    BvnComponent, GalaxyCacheComponent, GalaxySigmaDerivs,
    get_bvn_cov, galaxy_prototypes, linear_world_to_pix

import ..SensitiveFloats:
    SensitiveFloat, zero_sensitive_float, zero_sensitive_float_array,
    multiply_sfs!, add_scaled_sfs!, clear!

include("deterministic_vi_image_psf/sensitive_float_fft.jl")
include("deterministic_vi_image_psf/lanczos.jl")
include("deterministic_vi_image_psf/fsm_matrices.jl")
include("deterministic_vi_image_psf/elbo_image_psf.jl")

export elbo_likelihood_with_fft!, FSMSensitiveFloatMatrices, initialize_fsm_sf_matrices!

end