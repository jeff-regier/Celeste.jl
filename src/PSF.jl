using Base.Test
import SloanDigitalSkySurvey: SDSS, PSF

const field_dir =
  joinpath(Pkg.dir("SloanDigitalSkySurvey"), "dat", "sample_field")
const run_num = "003900"
const camcol_num = "6"
const field_num = "0269"


using ForwardDiff
using Optim
using PyPlot

const field_dir =
  joinpath(Pkg.dir("SloanDigitalSkySurvey"), "dat", "sample_field")
const run_num = "003900"
const camcol_num = "6"
const field_num = "0269"

using PSF.get_x_matrix_from_psf
using PSF.render_psf
using PSF.unwrap_parameters

raw_psf_comp =
  SDSS.load_psf_data(field_dir, run_num, camcol_num, field_num, 1);
pixel_keep = 10:42

psf = PSF.get_psf_at_point(10.0, 10.0, raw_psf_comp);
psf = psf[pixel_keep, pixel_keep];
psf_2 = PSF.get_psf_at_point(500.0, 500.0, raw_psf_comp);
psf_2 = psf_2[pixel_keep, pixel_keep];


x_mat = PSF.get_x_matrix_from_psf(psf);

verbose = true

psf_max = maximum(psf)
#matshow(psf, vmax=1.2 * psf_max); PyPlot.colorbar(); PyPlot.title("PSF")

K = 2

using Celeste
import ElboDeriv

bvn = ElboDeriv.BvnComponent(mu_vec[1], inv(sigma_vec[1]), weight_vec[1], true)
comp = ElboDeriv.GalaxyCacheComponent()
















# Why is NM the best?
# K=3 is much slower and not much better than K=2 by the looks of it.
opt_result, mu_vec, sigma_vec, weight_vec = PSF.fit_psf_gaussians(psf, K=K,
  optim_method=Optim.NelderMead(), verbose=true);



initial_par = opt_result.minimum

function evaluate_fit{T <: Number}(par::Vector{T})
  gmm_psf = render_psf(par, x_mat)
  local fit = sum((psf .- gmm_psf) .^ 2)
  if verbose && T == Float64
    println("-------------------")
    println("Fit: $fit")
    mu_vec, sigma_vec, weight_vec = unwrap_parameters(par)
    println(mu_vec)
    println(sigma_vec)
    println(weight_vec)
  end
  fit
end

# ForwardDiff reverses the arguments relative to what Optim expects.
evaluate_fit_g_rev! = ForwardDiff.gradient(evaluate_fit, mutates=true)
function evaluate_fit_g!(x::Vector{Float64}, output::Vector{Float64})
  evaluate_fit_g_rev!(output, x)
end

evaluate_fit_h_rev! = ForwardDiff.hessian(evaluate_fit, mutates=true)
function evaluate_fit_h!(x::Vector{Float64}, output::Matrix{Float64})
  evaluate_fit_h_rev!(output, x)
end

@time optim_result_newton =
  Optim.optimize(evaluate_fit, evaluate_fit_g!, evaluate_fit_h!,
                initial_par, method=Optim.Newton())

# @time optim_result_bfgs =
#   Optim.optimize(evaluate_fit, evaluate_fit_g!, evaluate_fit_h!,
#                 initial_par, method=Optim.BFGS())

@time optim_result_nelder =
    Optim.optimize(evaluate_fit, evaluate_fit_g!, evaluate_fit_h!,
                    initial_par, method=Optim.NelderMead())
h = ForwardDiff.hessian(evaluate_fit, initial_par)
maximum(eig(h)[1]) / minimum(eig(h)[1])

opt_result_noop = PSF.fit_psf_gaussians(psf, K=2, initial_par=opt_result.minimum,
  optim_method=Optim.NelderMead(), verbose=true, iterations=1);

# Why isn't this better?  It's because NM has no derivative information and so
# doesn't stay near the optimum.
opt_result_2 = fit_psf_gaussians(psf_2, K=2, initial_par=opt_result.minimum,
  optim_method=Optim.NelderMead(), verbose=true);



############# EM initialization
gmm, scale = PSF.fit_psf_gaussians_em(psf);

em_mu_vec = Array(Vector{Float64}, 2)
em_sigma_vec = Array(Matrix{Float64}, 2)
em_weight_vec = Array(Float64, 2)

for k=1:2
  em_mu_vec[k] = gmm.μ[k,:][:]
  em_sigma_vec[k] = inv(GaussianMixtures.precision(gmm.Σ[k]))
  em_weight_vec[k] = gmm.w[k] * scale
end
em_par = PSF.wrap_parameters(em_mu_vec, em_sigma_vec, em_weight_vec)
@time optim_result_newton_em =
  Optim.optimize(evaluate_fit, evaluate_fit_g!, evaluate_fit_h!,
                em_par, method=Optim.Newton())



# Coordinate ascent?

mu_vec = Array(Vector{Float64}, 2)
sigma_vec = Array(Matrix{Float64}, 2)
weight_vec = Array(Float64, 2)

opt_result1, mu1, sigma1, weight1 =
  PSF.fit_psf_gaussians(psf, K=1,
    optim_method=Optim.NelderMead(), verbose=true);
gmm_psf1 = render_psf(opt_result1.minimum, x_mat);

opt_result2, mu2, sigma2, weight2 =
  PSF.fit_psf_gaussians(psf - gmm_psf1, K=1,
    optim_method=Optim.NelderMead(), verbose=true);

mu_vec[1] = mu1[1]
mu_vec[2] = mu2[1]
sigma_vec[1] = sigma1[1]
sigma_vec[2] = sigma2[1]
weight_vec[1] = weight1[1]
weight_vec[2] = weight2[1]

initial_par = PSF.wrap_parameters(mu_vec, sigma_vec, weight_vec)

@time optim_result_newton =
  Optim.optimize(evaluate_fit, evaluate_fit_g!, evaluate_fit_h!,
                initial_par, method=Optim.Newton())















# opt_result = fit_psf_gaussians(psf, initial_par=opt_result.minimum, K=2,
#   optim_method=Optim.AcceleratedGradientDescent(), verbose=true, iterations=20);
#

unwrap_parameters(opt_result.minimum)
gmm_psf = render_psf(opt_result.minimum, x_mat);
matshow(gmm_psf, vmax=1.2 * psf_max); PyPlot.colorbar(); PyPlot.title("fit1")
psf_residual = psf - gmm_psf;
resid_max = 1.5 * maximum(abs(psf_residual))
matshow(psf_residual, vmax=resid_max, vmin=-resid_max)
PyPlot.colorbar(); PyPlot.title("residual")
sum(psf_residual .^ 2)

# Compare to EM
gmm, scale = PSF.fit_psf_gaussians_em(psf);
em_psf = [ PSF.evaluate_gmm(gmm, x_mat[i, j]')[1] for
           i=1:size(x_mat, 1), j=1:size(x_mat, 2)];
em_psf_residual = psf - em_psf;
matshow(em_psf_residual, vmax=resid_max, vmin=-resid_max)
PyPlot.colorbar(); PyPlot.title("em residual")
sum(em_psf_residual .^ 2)



mu_vec = Array(Vector{T}, K)
sigma_vec = Array(Matrix{T}, K)
weight_vec = zeros(T, K)

for k = 1:K
  mu_vec[k] = gmm.μ[1, :][:]
  sigma_vec[k] = sigma_chol' * sigma_chol + sigma_min
  weight_vec[k] = exp(par[offset + 6]) + weight_min
end


PyPlot.close("all")

psf_max = maximum(psf)
matshow(psf, vmax=1.2 * psf_max); PyPlot.colorbar(); PyPlot.title("PSF")

opt_result_1 = fit_psf_gaussians(psf);
unwrap_parameters(opt_result_1.minimum)
gmm_psf1 = render_psf(opt_result_1.minimum);
psf_residual1 = psf - gmm_psf1;
matshow(gmm_psf1, vmax=1.2 * psf_max); PyPlot.colorbar(); PyPlot.title("fit1")
resid_max = 1.5 * maximum(abs(psf_residual1))
matshow(psf_residual1, vmax=resid_max, vmin=-resid_max)
PyPlot.colorbar(); PyPlot.title("residual1")


opt_result_2 = fit_psf_gaussians(psf_residual1);
unwrap_parameters(opt_result_2.minimum)
gmm_psf2 = gmm_psf1 + render_psf(opt_result_2.minimum);
psf_residual2 = psf - gmm_psf2;
#matshow(gmm_psf2); PyPlot.colorbar(); PyPlot.title("fit2")
matshow(psf_residual2, vmax=resid_max, vmin=-resid_max)
PyPlot.colorbar(); PyPlot.title("residual2")

opt_result_3 = fit_psf_gaussians(psf_residual2);
unwrap_parameters(opt_result_3.minimum)
gmm_psf3 = gmm_psf2 + render_psf(opt_result_3.minimum);
psf_residual3 = psf - gmm_psf3;
matshow(psf_residual3, vmax=resid_max, vmin=-resid_max)
PyPlot.colorbar(); PyPlot.title("residual2")
