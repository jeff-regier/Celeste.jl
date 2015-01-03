# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module ElboDeriv

using CelesteTypes
import Planck
import Util


immutable BvnComponent
	the_mean::Vector{Float64}
	precision::Matrix{Float64}
	z::Float64

	BvnComponent(the_mean, the_cov, weight) = begin
		c = 1 ./ (det(the_cov)^.5 * 2pi)
		new(the_mean, the_cov^-1, c * weight)
	end
end


function ret_pdf(bmc::BvnComponent, x::Vector{Float64})
	y1 = x[1] - bmc.the_mean[1]
	y2 = x[2] - bmc.the_mean[2]
	py1 = bmc.precision[1,1] * y1 + bmc.precision[1,2] * y2
	py2 = bmc.precision[2,1] * y1 + bmc.precision[2,2] * y2
    c_ytpy = -0.5 * (y1 * py1 + y2 * py2)
	f_denorm = exp(c_ytpy)
    py1, py2, bmc.z * f_denorm
end


function accum_star!(bmc::BvnComponent, x::Vector{Float64},
		gamma_b::Float64, fs0m::SourceParam)
	py1, py2, f = ret_pdf(bmc, x)
	f_gamma = f * gamma_b

    fs0m.v += f_gamma
    fs0m.d[1] += f_gamma .* py1 #mu1
    fs0m.d[2] += f_gamma .* py2 #mu2
    fs0m.d[3] += f #gamma_b
end


function accum_galaxy!(bmc::BvnComponent, x::Vector{Float64},
		zeta_b::Float64, theta_i::Float64, theta_dir::Float64, 
		Xi::(Float64,Float64,Float64), st::Float64, fs1m::SourceParam)
	py1, py2, f_pre = ret_pdf(bmc, x)
	f = f_pre * zeta_b * theta_i

    fs1m.v += f
    fs1m.d[1] += f .* py1 #mu1
    fs1m.d[2] += f .* py2 #mu2
    fs1m.d[3] += f_pre * theta_i #theta
    fs1m.d[4] += theta_dir * f_pre * zeta_b #zeta

    df_dSigma_11 = 0.5 * f * (py1 * py1 - bmc.precision[1, 1])
	df_dSigma_12 = f * (py1 * py2 - bmc.precision[1, 2])  # NB: 2X
	df_dSigma_22 = 0.5 * f * (py2 * py2 - bmc.precision[2, 2])

    fs1m.d[5] += st * (df_dSigma_11 * 2Xi[1] + df_dSigma_12 * Xi[2])
    fs1m.d[6] += st * (df_dSigma_12 * Xi[1] + df_dSigma_22 * 2Xi[2])
    fs1m.d[7] += st * (df_dSigma_22 * 2Xi[3])
end


function elbo_likelihood(img::Image, V::VariationalParams)
	S = length(V)

    E_F = zero_all_matrix(img.H, img.W, S, full_index)
	fill!(E_F.v, img.epsilon)
    var_F = zero_all_matrix(img.H, img.W, S, full_index)

	const star_b_index = [2:3, 3 + img.b]
	fs0m = zero_source_param(star_b_index)

	const gal_b_index = [2:3, 9 + img.b, 15, 16:18]
	fs1m = zero_source_param(gal_b_index)

	for s in 1:S
		Vs = V[s]

		star_mcs = Array(BvnComponent, 3)
		for k in 1:3
			pc = img.psf[k]
			mean_s = [pc.xiBar[1] + Vs.mu[1], pc.xiBar[2] + Vs.mu[2]]
			star_mcs[k] = BvnComponent(mean_s, pc.SigmaBar, pc.alphaBar)
		end

		Xi = [[Vs.Xi[1] Vs.Xi[2]], [0. Vs.Xi[3]]]
		XiXi = Xi' * Xi

		gal_mcs = Array(BvnComponent, 3, 8, 2)
		for i = 1:2
			for j in 1:[6,8][i]
				gc = galaxy_prototypes[i][j]
				for k = 1:3
					pc = img.psf[k]
					mean_s = [pc.xiBar[1] + Vs.mu[1], pc.xiBar[2] + Vs.mu[2]]
					var_s = pc.SigmaBar + gc.sigmaTilde * XiXi
					weight = pc.alphaBar * gc.alphaTilde
					gal_mcs[k, j, i] = BvnComponent(mean_s, var_s, weight)
				end
			end
		end

		chi_var = (Vs.chi * (1. - Vs.chi))
		chi_var_d = 1. - 2 * Vs.chi

		for w in 1:img.W, h in 1:img.H
			m = Float64[h, w]

			clear_source_param!(fs0m)
			for star_mc in star_mcs
				accum_star!(star_mc, m, Vs.gamma[img.b], fs0m)
			end
	
			clear_source_param!(fs1m)
			for i = 1:2
				theta_dir = (i == 1) ? 1. : -1.
				theta_i = (i == 1) ? Vs.theta : 1. - Vs.theta

				for j in 1:[6,8][i]
					for k = 1:3
						accum_galaxy!(gal_mcs[k, j, i], m, Vs.zeta[img.b],
							theta_i, theta_dir, Vs.Xi, galaxy_prototypes[i][j].sigmaTilde, fs1m)
					end
				end
			end
			
			E_F.v[h, w] += (1. - Vs.chi) * fs0m.v + Vs.chi * fs1m.v

			E_F.d[1, h, w, s] += fs1m.v - fs0m.v  # 1 = chi
			for i in 1:length(fs0m.index)
				p = fs0m.index[i]
				E_F.d[p, h, w, s] += (1. - Vs.chi) * fs0m.d[i]
			end
			for i in 1:length(fs1m.index)
				p = fs1m.index[i]
				E_F.d[p, h, w, s] += Vs.chi * fs1m.d[i]
			end

			diff10 = fs1m.v - fs0m.v
			diff10_sq = diff10^2
			var_F.v[h, w] += chi_var * diff10_sq
			var_F.d[1, h, w, s] += chi_var_d * diff10_sq  # 1 = chi
			for i in 1:length(fs0m.index)
				p = fs0m.index[i]
				var_F.d[p, h, w, s] -= chi_var * 2 * diff10 * fs0m.d[i]
			end
			for i in 1:length(fs1m.index)
				p = fs1m.index[i]
				var_F.d[p, h, w, s] += chi_var * 2 * diff10 * fs1m.d[i]
			end
		end
	end

	ret_v = -sum(lfact(img.pixels))
	for w in 1:img.W, h in 1:img.H
		ret_v += img.pixels[h, w] * (log(E_F.v[h,w]) - 
				var_F.v[h,w] ./ (2. * (E_F.v[h, w]).^2))
		ret_v -= E_F.v[h, w]
	end
	P = get_dim(full_index)
	ret_d = zeros(P, S)
	for s in 1:S, w in 1:img.W, h in 1:img.H, p in 1:P
		ret_d[p, s] += img.pixels[h, w] * (E_F.d[p, h, w, s] / E_F.v[h,w] 
			- 0.5 * (E_F.v[h, w]^2 * var_F.d[p, h, w, s] - 
				var_F.v[h, w] * 2 * E_F.v[h, w] * E_F.d[p, h, w, s]) 
					./  E_F.v[h, w]^4)
		ret_d[p, s] -= E_F.d[p, h, w, s]
	end

    # sum(img.pixels .* E_log_F) - sum(E_F) -sum(lfact(img.pixels))
	AllParam(ret_v, ret_d, full_index)
end


function elbo_likelihood(blob::Blob, V::VariationalParams)
	ret = zero_all_param(length(V), full_index)
	for img in blob
		accum_all_param!(elbo_likelihood(img, V), ret)
	end
	ret
end


function regularizer(M::PriorParams, Vs::SourceParams)
	planck_colors = Planck.expected_colors(Vs.tau)
	gamma_colors = log(Vs.gamma[2:5]) - log(Vs.gamma[1:4])
	w_s = planck_colors - gamma_colors
	star_penalty = -.5 * (w_s' * M.Delta^-1 * w_s'')[]

	zeta_colors = log(Vs.zeta[2:5]) - log(Vs.zeta[1:4])
	z_s = M.Theta - zeta_colors
	galaxy_penalty = -.5 * (z_s' * M.Lambda^-1 * z_s'')[]

	(1. - Vs.chi) * star_penalty + Vs.chi * galaxy_penalty
end


function regularizer(M::PriorParams, V::VariationalParams)
	# could compute this faster by aggregating w_s and z_s before multiplication
	S = length(V)
	R = zero_all_param(S)
	for s in 1:S
		accum_new_source!(R, regularizer(M, V[s]), s)
	end
	R
end


function subtract_kl!(M::PriorParams, V::VariationalParams, accum::AllParam)
	for s in 1:length(V)
		Vs = V[s]

		accum.v -= Vs.chi * (log(Vs.chi) - log(M.rho))
		accum.v -= (1. - Vs.chi) * (log(1. - Vs.chi) - log(1. - M.rho))

		# 1 = chi
		accum.d[1, s] -= (log(Vs.chi) - log(M.rho)) + 1
		accum.d[1, s] -= -(log(1. - Vs.chi) - log(1. - M.rho)) - 1.
	end
end


function elbo(blob::Blob, M::PriorParams, V::VariationalParams)
	ret = elbo_likelihood(blob, V)# - kl(M, V) #+ regularizer(M, V) 
	#subtract_kl!(M, V, ret)
	ret
end


end

