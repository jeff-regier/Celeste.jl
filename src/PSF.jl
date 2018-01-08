"""
This module will be deleted shortly, when we beginning using a rasterized PSF.
In the future, PSF code will appear only in `SDSSIO.jl` (to load the eigen-PSF)
and `model/psf.jl` (functions for computing parts of the log probability
relating to the PSF).
"""
module PSF

using Celeste
using Celeste: Const, @aliasscope, @unroll_loop
using ..Model
using ..Model: GalaxySigmaDerivs, BivariateNormalDerivatives, get_bvn_derivs!,
               eval_bvn_pdf!, transform_bvn_derivs!, BvnComponent, get_bvn_cov
using ..Transform
using ..SensitiveFloats.SensitiveFloat
import ..SensitiveFloats

import Optim
import WCS

using ForwardDiff
using StaticArrays


const ID_MAT_2D = eye(Float64, 2)

"""
A data type to store functions related to optimizing a PSF fit. Initialize
using the transform and number of components, and then call fit_psf
to perform a fit to a specificed psf and initial parameters.
"""
mutable struct PsfOptimizer
    psf_transform::DataTransform
    ftol::Float64
    grtol::Float64
    num_iters::Int
    verbose::Bool
    K::Int

    # Variable that will be allocated in optimization:
    bvn_derivs::BivariateNormalDerivatives{Float64}

    log_pdf::SensitiveFloat{Float64}
    pdf::SensitiveFloat{Float64}
    pixel_value::SensitiveFloat{Float64}
    squared_error::SensitiveFloat{Float64}
    sf_free::SensitiveFloat{Float64}

    psf_params_free_vec_cache::Vector{Float64}

    function PsfOptimizer(psf_transform::DataTransform, K::Int;
                          verbose::Bool=false,
                          ftol::Float64=1e-9,
                          grtol::Float64=1e-9)
        num_iters = 50

        bvn_derivs = BivariateNormalDerivatives{Float64}()

        log_pdf = SensitiveFloat{Float64}(length(PsfParams), 1, true, true)
        pdf = SensitiveFloat{Float64}(length(PsfParams), 1, true, true)
        pixel_value = SensitiveFloat{Float64}(length(PsfParams), K, true, true)
        squared_error = SensitiveFloat{Float64}(length(PsfParams), K, true, true)
        sf_free = SensitiveFloat{Float64}(length(PsfParams), K, true, true)

        psf_params_free_vec_cache = fill(NaN, K * length(PsfParams))

        new(psf_transform, ftol, grtol, num_iters, verbose, K,
            bvn_derivs, log_pdf, pdf, pixel_value, squared_error,
            sf_free, psf_params_free_vec_cache)
    end
end

############################################################################################

function psf_fit_for_optim(psf_optimizer::PsfOptimizer, raw_psf, x_mat, psf_params_free_vec::Vector)
    if psf_params_free_vec == psf_optimizer.psf_params_free_vec_cache
        return psf_optimizer.sf_free
    else
        copy!(psf_optimizer.psf_params_free_vec_cache, psf_params_free_vec)
    end

    psf_params_free = unwrap_psf_params(psf_params_free_vec)
    psf_params = constrain_psf_params(psf_params_free, psf_optimizer.psf_transform)

    # Update squared_error in place.
    evaluate_psf_fit!(psf_params, raw_psf, x_mat, psf_optimizer.bvn_derivs,
                      psf_optimizer.log_pdf, psf_optimizer.pdf,
                      psf_optimizer.pixel_value, psf_optimizer.squared_error, true)

    # Update sf_free in place.
    transform_psf_sensitive_float!(psf_params, psf_optimizer.psf_transform,
                                   psf_optimizer.squared_error,
                                   psf_optimizer.sf_free, true)

    psf_optimizer.sf_free
end

function fit_psf(psf_optimizer::PsfOptimizer, raw_psf::Matrix{Float64}, initial_params::Vector{Vector{Float64}})
    x_mat = get_x_matrix_from_psf(raw_psf)
    psf_params_free = unconstrain_psf_params(initial_params, psf_optimizer.psf_transform)
    psf_params_free_vec = vec(wrap_psf_params(psf_params_free))

    function psf_fit_value(psf_params_free_vec::Vector)
        psf_fit_for_optim(psf_optimizer, raw_psf, x_mat, psf_params_free_vec).v[]
    end

    function psf_fit_grad!(psf_params_free_vec::Vector{Float64}, grad::Vector{Float64})
        grad[:] = psf_fit_for_optim(psf_optimizer, raw_psf, x_mat, psf_params_free_vec).d[:]
    end

    function psf_fit_hess!(psf_params_free_vec::Vector{Float64}, hess::Matrix{Float64})
        hess[:] = psf_fit_for_optim(psf_optimizer, raw_psf, x_mat, psf_params_free_vec).h
        hess[:] = 0.5 * (hess + hess')
    end

    tr_method = Optim.NewtonTrustRegion(initial_delta=10.0, delta_hat=1e9, eta=0.1,
                                        rho_lower=0.2, rho_upper=0.75)

    options = Optim.Options(;
                            x_tol = 0.0, # Don't allow convergence in params
                            f_tol = psf_optimizer.ftol,
                            g_tol = psf_optimizer.grtol,
                            iterations = psf_optimizer.num_iters,
                            store_trace = psf_optimizer.verbose,
                            show_trace = false,
                            extended_trace = psf_optimizer.verbose)

    return Optim.optimize(psf_fit_value, psf_fit_grad!, psf_fit_hess!,
                          psf_params_free_vec, tr_method, options)
end

############################################################################################

"""
Return an image of a Celeste GMM PSF evaluated at rows, cols.

Args:
- psf_array: The PSF to be evaluated as an array of PsfComponent
- rows: The rows in the image (usually in pixel coordinates)
- cols: The column in the image (usually in pixel coordinates)

Returns:
 - The PSF values at rows and cols.    The default size is the same as
     that returned by get_psf_at_point applied to FITS header values.

Note that the point in the image at which the PSF is evaluated --
that is, the center of the image returned by this function -- is
already implicit in the value of psf_array.
"""
function get_psf_at_point(psf_array::Vector{PsfComponent};
                          rows = -25:25, cols = -25:25)
    function get_psf_value(psf::PsfComponent, row::Float64, col::Float64)
         x = @SVector(Float64[row, col]) - psf.xiBar
         exp_term = exp(-0.5 * dot(x, psf.tauBarInv * x) - 0.5 * psf.tauBarLd)
         psf.alphaBar * exp_term / (2 * pi)
    end

    return Float64[
     sum(get_psf_value(psf, float(row), float(col)) for psf in psf_array)
         for row in rows, col in cols ]
end


"""
Get the PSF located at a particular world location in an image.

Args:
 - world_loc: A location in world coordinates.
 - img: An Image

Returns:
 - An array of PsfComponent objects that represents the PSF as a mixture
     of Gaussians.
"""
function get_source_psf(world_loc::Vector{Float64}, img::Image, psf_K::Int)
    pixel_loc = WCS.world_to_pix(img.wcs, world_loc)
    psfstamp = img.psfmap(pixel_loc[1], pixel_loc[2])
    return fit_raw_psf_for_celeste(psfstamp, psf_K)
end


"""
Get the PSF located at a particular world location in an image.

Args:
 - world_loc: A location in world coordinates.
 - img: An Image

Returns:
 - An array of PsfComponent objects that represents the PSF as a mixture
     of Gaussians.
"""
function get_source_psf(world_loc::Vector{Float64},
                        img::Image,
                        psf_optimizer::PsfOptimizer,
                        initial_psf_params::Vector{Vector{Float64}})
    pixel_loc = WCS.world_to_pix(img.wcs, world_loc)
    psfstamp = img.psfmap(pixel_loc[1], pixel_loc[2])
    return fit_raw_psf_for_celeste(psfstamp, psf_optimizer, initial_psf_params)
end


"""
Get initial values for a set of PSF parameters.

Args:
    - K: The number of components.
    - for_test: If true, return a set of slightly asymmetric parameters so that
                            derivatives are interesting for testing.

Returns:
    - A vector of PSF parameters, one for each component.
"""
function initialize_psf_params(K::Int; for_test::Bool=false)
    psf_params = Vector{Vector{Float64}}(K)
    for k=1:K
        if for_test
            # Choose asymmetric values for testing.
            psf_params[k] = zeros(length(PsfParams))
            psf_params[k][psf_ids.mu] = [0.1, 0.2]
            psf_params[k][psf_ids.gal_axis_ratio] = 0.8
            psf_params[k][psf_ids.gal_angle] = pi / 4
            psf_params[k][psf_ids.gal_radius_px] = sqrt(2 * k)
            psf_params[k][psf_ids.weight] = 1 / K + k / 10
        else
            psf_params[k] = zeros(length(PsfParams))
            psf_params[k][psf_ids.mu] = [0.0, 0.0]
            psf_params[k][psf_ids.gal_axis_ratio] = 0.95
            psf_params[k][psf_ids.gal_angle] = 0.0
            psf_params[k][psf_ids.gal_radius_px] = sqrt(2 * k)
            psf_params[k][psf_ids.weight] = 1 / K
        end
    end

    psf_params
end


"""
Given PSF parameters, return a DataTransform to transfrom them to and from
an unconstrained parameterization.

Args:
    - psf_params: The parameters to be transformed
    - scale: A vector as long as PsfParams to define a linear rescaling.

Returns:
    - A DataTransform object.
"""
function get_psf_transform(
        psf_params::Vector{Vector{Float64}};
        scale::Vector{Float64}=ones(length(PsfParams)))
    K = length(psf_params)
    bounds = Vector{ParamBounds}(length(psf_params))
    # Note that, for numerical reasons, the bounds must be on the scale
    # of reasonably meaningful changes.
    for k in 1:K
        bounds[k] = ParamBounds()
        bounds[k][:mu] = ParamBox[ ParamBox(-5.0, 5.0, scale[psf_ids.mu[1]]),
                                   ParamBox(-5.0, 5.0, scale[psf_ids.mu[2]]) ]
        bounds[k][:gal_axis_ratio] = ParamBox[ ParamBox(0.1, 1.0, scale[psf_ids.gal_axis_ratio] ) ]
        bounds[k][:gal_angle] =
            ParamBox[ ParamBox(-4 * pi, 4 * pi, scale[psf_ids.gal_angle] ) ]
        bounds[k][:gal_radius_px] =
            ParamBox[ ParamBox(0.05, 10.0, scale[psf_ids.gal_radius_px] ) ]

        # Note that the weights do not need to sum to one.
        bounds[k][:weight] = ParamBox[ ParamBox(0.05, 2.0, scale[psf_ids.weight] ) ]
    end
    DataTransform(bounds, collect(1:K), K)
end


"""
Transform psf_params to and from psf_params_free in place.

Args:
    - psf_params, psf_params_free: The constrained and unconstrained parameters,
            respectively.
    - psf_transform: The DataTransform to be applied.
    - to_unconstrained: If true, then update psf_params_free with the unconstrained
            representation of psf_params.    If false, update psf_params with the
            constrained version of psf_params_free.

Returns:
    - Updates psf_params or psf_params_free in place.
"""
function transform_psf_params!(psf_params::Vector{Vector{T}},
                               psf_params_free::Vector{Vector{T}},
                               psf_transform::DataTransform,
                               to_unconstrained::Bool) where {T<:Number}
    for k=1:length(psf_params)
        for (param, constraint_vec) in psf_transform.bounds[k]
            for ind in 1:length(getfield(psf_ids, param))
                param_ind = getfield(psf_ids, param)[ind]
                constraint = constraint_vec[ind]
                to_unconstrained ?
                    psf_params_free[k][param_ind] =
                        Transform.unbox_parameter(psf_params[k][param_ind], constraint):
                    psf_params[k][param_ind] =
                        Transform.box_parameter(psf_params_free[k][param_ind], constraint)
            end
        end
    end

    true # return type
end


"""
Allocate memory for and return a constrained parameter set.
"""
function constrain_psf_params(psf_params_free::Vector{Vector{T}},
                              psf_transform::DataTransform) where {T<:Number}
    K = length(psf_params_free)
    psf_params = Vector{Vector{T}}(K)
    for k=1:K
        psf_params[k] = zeros(T, length(PsfParams))
    end

    transform_psf_params!(psf_params, psf_params_free, psf_transform, false)

    psf_params
end


"""
Allocate memory for and return an unconstrained parameter set.
"""
function unconstrain_psf_params(psf_params::Vector{Vector{T}},
                                psf_transform::DataTransform) where {T<:Number}
    K = length(psf_params)
    psf_params_free = Vector{Vector{T}}(K)
    for k=1:K
        psf_params_free[k] = zeros(T, length(PsfParams))
    end

    transform_psf_params!(psf_params, psf_params_free, psf_transform, true)

    psf_params_free
end


"""
Convert a single vector of psf parameters to a vector of vectors.
"""
function unwrap_psf_params(psf_param_vec::Vector{T}) where {T<:Number}
    @assert length(psf_param_vec) % length(PsfParams) == 0
    K = round(Int, length(psf_param_vec) / length(PsfParams))
    psf_param_mat = reshape(psf_param_vec, length(PsfParams), K)
    psf_params = Vector{Vector{T}}(K)
    for k = 1:K
        psf_params[k] = psf_param_mat[:, k]
    end
    psf_params
end


"""
Convert a vector of vectors of psf parameters to a single vector.
"""
function wrap_psf_params(psf_params::Vector{Vector{T}}) where {T<:Number}
    psf_params_mat = zeros(T, length(PsfParams), length(psf_params))
    for k=1:length(psf_params)
        psf_params_mat[:, k] = psf_params[k]
    end
    psf_params_mat[:]
end


"""
Return a sensitive float representing the value of the psf at pixel x
with all its associated derivatives (with respect to the constrained
parameterization).    Note that "fit" is a bad name -- it is just the pixel value.
TODO: fix the name.

Args:
    - x: The 2d location at which to evaluate the pdf
    - calculate_gradient: If false, only update the value of pixel_value
    - other values: Pre-allocated memory for intermediate calculations

Returns:
    - Updates pixel_value in place (and all the other placeholder values as well)
"""
function evaluate_psf_pixel_fit!(
        x::SVector{2,Float64},
        psf_params::Vector{Vector{T}},
        sig_sf_vec::Vector{GalaxySigmaDerivs{T}},
        bvn_vec::Vector{BvnComponent{T}},
        bvn_derivs::BivariateNormalDerivatives{T},
        log_pdf::SensitiveFloat{T},
        pdf::SensitiveFloat{T},
        pixel_value::SensitiveFloat{T},
        calculate_gradient::Bool) where {T<:Number}
    SensitiveFloats.zero!(pixel_value)

    K = length(psf_params)
    sigma_ids = (psf_ids.gal_axis_ratio, psf_ids.gal_angle, psf_ids.gal_radius_px)
    @inbounds for k = 1:K
        # I will put in the weights later so that the log pdf sensitive float
        # is accurate.
        bvn = bvn_vec[k]
        eval_bvn_pdf!(bvn_derivs, bvn, x)
        get_bvn_derivs!(bvn_derivs, bvn, true, true)
        transform_bvn_derivs!(bvn_derivs, sig_sf_vec[k], I, true)

        SensitiveFloats.zero!(log_pdf)
        SensitiveFloats.zero!(pdf)

        # This is redundant, but it's what eval_bvn_pdf returns.
        log_pdf.v[] = log(bvn_derivs.f_pre[1])

        if calculate_gradient
            for ind=1:2
                log_pdf.d[psf_ids.mu[ind]] = bvn_derivs.bvn_u_d[ind]
            end
            for ind=1:3
                log_pdf.d[sigma_ids[ind]] = bvn_derivs.bvn_s_d[ind]
            end
            log_pdf.d[psf_ids.weight] = 0

            for ind1 = 1:2, ind2 = 1:2
                log_pdf.h[psf_ids.mu[ind1], psf_ids.mu[ind2]] =
                    bvn_derivs.bvn_uu_h[ind1, ind2]
            end
            for mu_ind = 1:2, sig_ind = 1:3
                log_pdf.h[psf_ids.mu[mu_ind], sigma_ids[sig_ind]] =
                log_pdf.h[sigma_ids[sig_ind], psf_ids.mu[mu_ind]] =
                    bvn_derivs.bvn_us_h[mu_ind, sig_ind]
            end
            for ind1 = 1:3, ind2 = 1:3
                log_pdf.h[sigma_ids[ind1], sigma_ids[ind2]] =
                    bvn_derivs.bvn_ss_h[ind1, ind2]
            end
        end

        pdf_val = exp(log_pdf.v[])
        pdf.v[] = pdf_val

        if calculate_gradient
            for ind1 = 1:length(PsfParams)
                if ind1 == psf_ids.weight
                    pdf.d[ind1] = pdf_val
                else
                    pdf.d[ind1] = psf_params[k][psf_ids.weight] * pdf_val * log_pdf.d[ind1]
                end

                for ind2 = 1:ind1
                    pdf.h[ind1, ind2] = pdf.h[ind2, ind1] =
                        psf_params[k][psf_ids.weight] * pdf_val *
                        (log_pdf.h[ind1, ind2] + log_pdf.d[ind1] * log_pdf.d[ind2])
                end
            end

            # Weight hessian terms.
            for ind1 = 1:length(PsfParams)
                pdf.h[psf_ids.weight, ind1] = pdf.h[ind1, psf_ids.weight] =
                    pdf_val * log_pdf.d[ind1]
            end

        end

        pdf.v[] *= psf_params[k][psf_ids.weight]

        SensitiveFloats.add_sources_sf!(pixel_value, pdf, k)
    end

    true # Set return type
end


"""
Convert PSF parameters to covariance matrices and derivatives and BvnComponents.
"""
function get_sigma_from_params(psf_params::Vector{Vector{T}}) where {T<:Number}
    K = length(psf_params)
    sigma_vec = Vector{SMatrix{2,2,T,4}}(K)
    sig_sf_vec = Vector{GalaxySigmaDerivs{T}}(K)
    bvn_vec = Vector{BvnComponent{T}}(K)
    for k = 1:K
        sigma_vec[k] = get_bvn_cov(psf_params[k][psf_ids.gal_axis_ratio],
            psf_params[k][psf_ids.gal_angle],
            psf_params[k][psf_ids.gal_radius_px])
        sig_sf_vec[k] = GalaxySigmaDerivs(
            psf_params[k][psf_ids.gal_angle],
            psf_params[k][psf_ids.gal_axis_ratio],
            psf_params[k][psf_ids.gal_radius_px], sigma_vec[k], 1.0, true)

        bvn_vec[k] =
            BvnComponent(SVector{2,T}(psf_params[k][psf_ids.mu]), sigma_vec[k], 1.0)
    end
    sigma_vec, sig_sf_vec, bvn_vec
end


"""
evaluate_psf_fit but with pre-allocated memory for intermediate calculations.
"""
function evaluate_psf_fit!(
        psf_params::Vector{Vector{T}},
        raw_psf::Matrix{Float64},
        x_mat::Matrix{SVector{2,Float64}},
        bvn_derivs::BivariateNormalDerivatives{T},
        log_pdf::SensitiveFloat{T},
        pdf::SensitiveFloat{T},
        pixel_value::SensitiveFloat{T},
        squared_error::SensitiveFloat{T},
        calculate_gradient::Bool) where {T<:Number}
    K = length(psf_params)
    sigma_vec, sig_sf_vec, bvn_vec = get_sigma_from_params(psf_params)
    SensitiveFloats.zero!(squared_error)

    @inbounds for x_ind in 1:length(x_mat)
        SensitiveFloats.zero!(pixel_value)
        evaluate_psf_pixel_fit!(
                x_mat[x_ind], psf_params, sig_sf_vec, bvn_vec,
                bvn_derivs, log_pdf, pdf, pixel_value, calculate_gradient)

        diff = (pixel_value.v[] - raw_psf[x_ind])
        squared_error.v[] +=    diff ^ 2
        if calculate_gradient
            for ind1 = 1:length(squared_error.d)
                squared_error.d[ind1] += 2 * diff * pixel_value.d[ind1]
                for ind2 = 1:ind1
                    squared_error.h[ind1, ind2] +=
                        2 * (diff * pixel_value.h[ind1, ind2] +
                                 pixel_value.d[ind1] * pixel_value.d[ind2]')
                    squared_error.h[ind2, ind1] = squared_error.h[ind1, ind2]
                end
            end
        end # if calculate_gradient
    end

    squared_error
end


"""
Transform a sf, a SensitiveFloat that has derivatives with respect to
constrianed parameters, into sf_free, a SensitiveFloat that has derivatives
with respect to unconstrained parameteres.

Args:
    - psf_params: The constrianed parameters
    - psf_transform: The data transform to be applied
    - sf: The SensitiveFloat with derivatives with respect to the constrained params
    - sf_free: Updated in place.    The SensitiveFloat with derivatives with respect
                         to the unconstrained parameters.
    - calculate_gradient: If false, only calculate the value.

Returns:
    - Updates sf_free in place.
"""
function transform_psf_sensitive_float!(
        psf_params::Vector{Vector{T}},
        psf_transform::DataTransform,
        sf::SensitiveFloat{T},
        sf_free::SensitiveFloat{T},
        calculate_gradient::Bool) where {T<:Number}
    sf_free.v[] = sf.v[]
    if calculate_gradient
        K = length(psf_params)

        # These are the hessians of each individual parameter's transform.    We
        # can represent it this way since each parameter's transform only depends on
        # its own value and not on others.

        # This is the diagonal of the Jacobian transform.
        jacobian_diag = zeros(length(PsfParams) * K)

        # These are hte Hessians of the each parameter's transform.
        hessian_values = zeros(length(PsfParams) * K)

        for k = 1:K
            offset = length(PsfParams) * (k - 1)
            for ind = 1:2
                mu_ind = psf_ids.mu[ind]
                jac, hess = Transform.box_derivatives(
                    psf_params[k][mu_ind], psf_transform.bounds[k][:mu][ind])
                jacobian_diag[offset + mu_ind] = jac
                hessian_values[offset + mu_ind] = hess
            end

            # The rest are one-dimensional.
            for field in setdiff(fieldnames(PsfParams), [ :mu ])
                ind = getfield(psf_ids, field)
                jac, hess = Transform.box_derivatives(
                    psf_params[k][ind], psf_transform.bounds[k][field][1])
                jacobian_diag[offset + ind] = jac
                hessian_values[offset + ind] = hess
            end
        end

        # Apply the transformations.
        for ind1 = 1:length(sf_free.d)
            sf_free.d[ind1] = jacobian_diag[ind1] * sf.d[ind1]

            for ind2 = 1:ind1
                # Calculate the Hessian
                sf_free.h[ind1, ind2] = sf_free.h[ind2, ind1] =
                    (jacobian_diag[ind1] * jacobian_diag[ind2]) * sf.h[ind1, ind2]
                if ind1 == ind2
                    sf_free.h[ind1, ind2] +=    hessian_values[ind1] * sf.d[ind1]
                end
            end
        end
    end

    true # return type
end


"""
Given a psf matrix, return a matrix of the same size, where each element of
the matrix is a 2-length vector of the [x1, x2] location of that matrix cell.
The locations are chosen so that the scale is pixels, but the center of the
psf is [0, 0].
"""
function get_x_matrix_from_psf(psf::Matrix{Float64})
    psf_center1, psf_center2 = (size(psf, 1) - 1) / 2 + 1, (size(psf, 2) - 1) / 2 + 1
    return [ @SVector Float64[i - psf_center1, j - psf_center2] for i = 1:size(psf, 1), j = 1:size(psf, 2) ]
end


"""
Fit a mixture of 2d Gaussians to a PSF image (evaluated at a single point).

Args:
    - raw_psf: An matrix image of the point spread function.
    - psf_optimizer: A PsfOptimizer object for the fit.

Returns:
    - A vector of PsfComponents fit to the raw_psf.
"""
function fit_raw_psf_for_celeste(raw_psf::Array{Float64, 2},
                                 psf_optimizer::PsfOptimizer,
                                 initial_psf_params::Vector{Vector{Float64}})
    K = length(initial_psf_params)
    @assert K == psf_optimizer.K
    optim_result = fit_psf(psf_optimizer, raw_psf, initial_psf_params)
    psf_params_fit =
        constrain_psf_params(
            unwrap_psf_params(Optim.minimizer(optim_result)), psf_optimizer.psf_transform)

    sigma_vec = get_sigma_from_params(psf_params_fit)[1]
    celeste_psf = Vector{PsfComponent}(K)
    for k=1:K
        mu = psf_params_fit[k][psf_ids.mu]
        weight = psf_params_fit[k][psf_ids.weight]
        celeste_psf[k] = PsfComponent(weight, SVector{2,eltype(mu)}(mu), SMatrix{2,2,eltype(mu),4}(sigma_vec[k]))
    end

    celeste_psf, psf_params_fit
end


"""
Fit a mixture of 2d Gaussians to a PSF image (evaluated at a single point).

Args:
 - raw_psf: An matrix image of the point spread function.
 - K: The number of components to fit.

 Returns:
     - A vector of PsfComponents fit to the raw_psf.
"""
function fit_raw_psf_for_celeste(raw_psf::Array{Float64, 2}, K::Integer; ftol=1e-9)
    psf_params = initialize_psf_params(K, for_test=false)
    psf_transform = get_psf_transform(psf_params)
    psf_optimizer = PsfOptimizer(psf_transform, K)
    psf_optimizer.ftol = ftol
    fit_raw_psf_for_celeste(raw_psf, psf_optimizer, psf_params)
end


function trim_psf(raw_psf::Array{Float64, 2}; trim_percent=0.999)
    h_mid = cld(size(raw_psf, 1), 2)
    w_mid = cld(size(raw_psf, 2), 2)

    width = 1
    function get_trimmed_psf_image()
        raw_psf[(h_mid - width):(h_mid + width),
                (w_mid - width):(w_mid + width)]
    end

    psf_tot = sum(abs, raw_psf)
    while sum(abs, get_trimmed_psf_image()) < trim_percent * psf_tot
        width += 1
    end

    deepcopy(get_trimmed_psf_image())
end

end # module
