

function lanczos_kernel{NumType <: Number}(x::NumType, a::Float64)
    abs(x) < a ? sinc(x) * sinc(x / a): zero(NumType)
end


function sinc_with_derivatives{NumType <: Number}(x::NumType)
    x_pi = pi * x
    sinc_x = sinc(x)
    sinc_x_d = (cos(x_pi) - sinc_x) / x
    sinc_x_h = -pi * (pi * sinc_x + 2 * sinc_x_d / x_pi)
    return sinc_x, sinc_x_d, sinc_x_h
end


# A function without checking a.  Factored out for testing with ForwardDiff..
function lanczos_kernel_with_derivatives_nocheck{NumType <: Number}(
    x::NumType, a::Float64)
    sinc_x, sinc_x_d, sinc_x_h = sinc_with_derivatives(x)
    sinc_xa, sinc_xa_d, sinc_xa_h = sinc_with_derivatives(x / a)

    return sinc_x * sinc_xa,
           sinc_x_d * sinc_xa + sinc_x * sinc_xa_d / a,
           sinc_x_h * sinc_xa + 2 * sinc_x_d * sinc_xa_d / a +
              sinc_x * sinc_xa_h / (a ^ 2)
end


function lanczos_kernel_with_derivatives{NumType <: Number}(x::NumType, a::Float64)
    if abs(x) > a
        return 0, 0, 0
    end
    return lanczos_kernel_with_derivatives_nocheck(x, a)
end


# Interpolate the PSF to the pixel values.
function lanczos_interpolate!{NumType <: Number, ParamType <: ParamSet}(
        image::Matrix{SensitiveFloat{ParamType, NumType}},
        psf_image::Matrix{Float64},
        object_loc::Vector{NumType},
        lanczos_width::Int,
        wcs_jacobian::Matrix{Float64},
        calculate_derivs::Bool,
        calculate_hessian::Bool)

    a = Float64(lanczos_width)
    h_psf_width = (size(psf_image, 1) + 1) / 2.0
    w_psf_width = (size(psf_image, 2) + 1) / 2.0

    param_ids = getids(ParamType)

    # These are sensitive floats representing derviatives of the Lanczos kernel.
    kernel = zero_sensitive_float(ParamType, NumType, 1)
    kernel_h = zero_sensitive_float(ParamType, NumType, 1)

    # Pre-compute terms for transforming derivatives to world coordinates.
    if calculate_derivs
        k_h_grad = wcs_jacobian' * Float64[1, 0]
        k_w_grad = wcs_jacobian' * Float64[0, 1]

        if calculate_hessian
            k_h_hess = wcs_jacobian' * Float64[1 0; 0 0] * wcs_jacobian
            k_w_hess = wcs_jacobian' * Float64[0 0; 0 1] * wcs_jacobian
        end
    end

    # h, w are pixel coordinates.
    for h = 1:size(image, 1), w = 1:size(image, 2)

        # h_psf, w_psf are in psf coordinates.
        # The PSF is centered at object_loc + psf_width.
        h_psf = h - object_loc[1] + h_psf_width
        w_psf = w - object_loc[2] + w_psf_width

        # Centers of indices of the psf matrix, i.e., integer psf coordinates.
        h_ind0, w_ind0 = Int(floor(h_psf)), Int(floor(w_psf))
        h_lower = max(h_ind0 - lanczos_width + 1, 1)
        h_upper = min(h_ind0 + lanczos_width, size(psf_image, 1))
        for h_ind = (h_lower:h_upper)
            lh_v, lh_d, lh_h = lanczos_kernel_with_derivatives(h_psf - h_ind, a)
            if lh_v != 0
                clear!(kernel_h)
                kernel_h.v[1] = lh_v

                if calculate_derivs
                    # This is -1 * wcs_jacobian' * [lh_d, 0]
                    # and -1 * wcs_jacobian' * [lh_h 0; 0 0] * wcs_jacobian
                    kernel_h.d[param_ids.u] = -1 * k_h_grad * lh_d
                    if calculate_hessian
                        kernel_h.h[param_ids.u, param_ids.u] = k_h_hess * lh_h;
                    end
                end
                w_lower = max(w_ind0 - lanczos_width + 1, 1)
                w_upper = min(w_ind0 + lanczos_width, size(psf_image, 2))
                for w_ind = (w_lower:w_upper)
                    lw_v, lw_d, lw_h =
                        lanczos_kernel_with_derivatives(w_psf - w_ind, a)
                    if lw_v != 0
                        clear!(kernel)
                        kernel.v[1] = lw_v
                        # This is -1 * wcs_jacobian' * [0, lw_d]
                        # and -1 * wcs_jacobian' * [0 0; 0 lw_h] * wcs_jacobian
                        if calculate_derivs
                            kernel.d[param_ids.u] = -1 * k_w_grad * lw_d;
                            if calculate_hessian
                                kernel.h[param_ids.u, param_ids.u] = k_w_hess * lw_h;
                            end
                            multiply_sfs!(kernel, kernel_h, calculate_hessian)
                            add_scaled_sfs!(
                                image[h, w], kernel, psf_image[h_ind, w_ind],
                                calculate_hessian)
                        else
                            image[h, w].v[1] +=
                                kernel.v[1] * kernel_h.v[1] * psf_image[h_ind, w_ind]
                        end
                    end
                end
            end
        end
    end
end
