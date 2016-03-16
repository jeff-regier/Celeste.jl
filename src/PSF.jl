using Celeste
using Celeste.Types
using Celeste.SkyImages
using Celeste.BivariateNormals
import Celeste.Util
import Celeste.SensitiveFloats
import Celeste.SDSSIO

using SloanDigitalSkySurvey.PSF
using Celeste.SensitiveFloats.SensitiveFloat
using Celeste.SensitiveFloats.clear!


function evaluate_psf_pixel_fit!{NumType <: Number}(
    x::Vector{Float64}, psf_params::Matrix{NumType},
    sigma_vec::Vector{Matrix{NumType}},
    sig_sf_vec::Vector{GalaxySigmaDerivs{NumType}},
    bvn_derivs::BivariateNormalDerivatives{NumType},
    log_pdf::SensitiveFloat{PsfParams, NumType},
    pdf::SensitiveFloat{PsfParams, NumType},
    pixel_value::SensitiveFloat{PsfParams, NumType})

  clear!(pixel_value)

  sigma_ids = [psf_ids.e_axis, psf_ids.e_angle, psf_ids.e_scale]
  for k = 1:K
    # I will put in the weights later so that the log pdf sensitive float
    # is accurate.
    bvn = BvnComponent{NumType}(psf_params[psf_ids.mu, k], sigma_vec[k], 1.0);
    eval_bvn_pdf!(bvn_derivs, bvn, x)
    get_bvn_derivs!(bvn_derivs, bvn, true, true)
    transform_bvn_derivs!(bvn_derivs, sig_sf_vec[k], eye(Float64, 2), true)

    clear!(log_pdf)
    clear!(pdf)

    # This is redundant, but it's what eval_bvn_pdf returns.
    log_pdf.v[1] = log(bvn_derivs.f_pre[1])
    log_pdf.d[psf_ids.mu] = bvn_derivs.bvn_u_d
    log_pdf.d[sigma_ids] = bvn_derivs.bvn_s_d
    log_pdf.d[psf_ids.weight] = 0

    log_pdf.h[psf_ids.mu, psf_ids.mu] = bvn_derivs.bvn_uu_h
    log_pdf.h[psf_ids.mu, sigma_ids] = bvn_derivs.bvn_us_h
    log_pdf.h[sigma_ids, psf_ids.mu] = log_pdf.h[psf_ids.mu, sigma_ids]'
    log_pdf.h[sigma_ids, sigma_ids] = bvn_derivs.bvn_ss_h

    pdf_val = exp(log_pdf.v[1])
    pdf.v[1] = pdf_val
    pdf.d = pdf_val * log_pdf.d
    pdf.h = pdf_val * (log_pdf.h + log_pdf.d * log_pdf.d')

    # Now multiply by the weight.
    pdf.h *= psf_params[psf_ids.weight, k]
    pdf.h[psf_ids.weight, :] = pdf.h[:, psf_ids.weight] = pdf.d

    pdf.d *= psf_params[psf_ids.weight, k]
    pdf.d[psf_ids.weight] = pdf_val

    pdf.v *= psf_params[psf_ids.weight, k]

    SensitiveFloats.add_sources_sf!(pixel_value, pdf, k, true)
  end

  true # Set return type
end


function evaluate_psf_fit{NumType <: Number}(
    psf_params::Matrix{NumType}, raw_psf::Matrix{Float64})

  K = size(psf_params, 2)

  #psf_image = zeros(size(x_mat));

  # TODO: allocate these outside?
  bvn_derivs = BivariateNormalDerivatives{Float64}(NumType);
  log_pdf = SensitiveFloats.zero_sensitive_float(PsfParams, NumType, 1);
  pdf = SensitiveFloats.zero_sensitive_float(PsfParams, NumType, 1);

  pixel_value = SensitiveFloats.zero_sensitive_float(PsfParams, NumType, K);
  squared_error = SensitiveFloats.zero_sensitive_float(PsfParams, NumType, K);

  sigma_vec = Array(Matrix{NumType}, K);
  sig_sf_vec = Array(GalaxySigmaDerivs{NumType}, K);

  for k = 1:K
    sigma_vec[k] = Util.get_bvn_cov(psf_params[psf_ids.e_axis, k],
                                    psf_params[psf_ids.e_angle, k],
                                    psf_params[psf_ids.e_scale, k])
    sig_sf_vec[k] = GalaxySigmaDerivs(
      psf_params[psf_ids.e_angle, k],
      psf_params[psf_ids.e_axis, k],
      psf_params[psf_ids.e_scale, k], sigma_vec[k], calculate_tensor=true);

  end

  SensitiveFloats.clear!(squared_error)

  for x_ind in 1:length(x_mat)
    evaluate_psf_pixel_fit!(
        x_mat[x_ind], psf_params, sigma_vec, sig_sf_vec,
        bvn_derivs, log_pdf, pdf, pixel_value)

    #psf_image[x_ind] = pixel_value.v[1]
    squared_error.v += (pixel_value.v[1] - raw_psf[x_ind]) ^ 2
    squared_error.d += 2 * (pixel_value.v[1] - raw_psf[x_ind]) * pixel_value.d
    squared_error.h += 2 * pixel_value.h
  end

  squared_error
end
