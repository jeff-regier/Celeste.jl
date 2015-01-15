# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module ElboDeriv

using CelesteTypes
import Util


immutable BvnComponent
	the_mean::Vector{Float64}
	precision::Matrix{Float64}
	z::Float64

	BvnComponent(the_mean, the_cov, weight) = begin
		the_det = the_cov[1,1] * the_cov[2,2] - the_cov[1,2] * the_cov[2,1]
		c = 1 ./ (the_det^.5 * 2pi)
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
		gamma_b::Float64, fs0m::SensitiveFloat)
	py1, py2, f = ret_pdf(bmc, x)
	f_gamma = f * gamma_b

    fs0m.v += f_gamma
    fs0m.d[1] += f_gamma .* py1 #mu1
    fs0m.d[2] += f_gamma .* py2 #mu2
    fs0m.d[3] += f #gamma_b
end


function accum_galaxy!(bmc::BvnComponent, x::Vector{Float64},
		zeta_b::Float64, theta_i::Float64, theta_dir::Float64, 
		Xi::(Float64,Float64,Float64), st::Float64, fs1m::SensitiveFloat)
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


function load_bvn_mixtures(psf::Vector{PsfComponent}, mp::ModelParams)
	star_mcs = Array(BvnComponent, 3, mp.S)
	gal_mcs = Array(BvnComponent, 3, 8, 2, mp.S)

	for s in 1:mp.S
		vs = mp.vp[s]

		for k in 1:3
			pc = psf[k]
			mean_s = [pc.xiBar[1] + vs.mu[1], pc.xiBar[2] + vs.mu[2]]
			star_mcs[k, s] = BvnComponent(mean_s, pc.SigmaBar, pc.alphaBar)
		end

		Xi = [[vs.Xi[1] vs.Xi[2]], [0. vs.Xi[3]]]
		XiXi = Xi' * Xi

		for i = 1:2
			for j in 1:[6,8][i]
				gc = galaxy_prototypes[i][j]
				for k = 1:3
					pc = psf[k]
					mean_s = [pc.xiBar[1] + vs.mu[1], pc.xiBar[2] + vs.mu[2]]
					var_s = pc.SigmaBar + gc.sigmaTilde * XiXi
					weight = pc.alphaBar * gc.alphaTilde
					gal_mcs[k, j, i, s] = BvnComponent(mean_s, var_s, weight)
				end
			end
		end
	end

	star_mcs, gal_mcs
end


function accum_pixel_source_stats!(star_mcs::Array{BvnComponent, 2}, 
		gal_mcs::Array{BvnComponent, 4}, vs::Vector{Float64}, s::Int64, 
		m_pos::Vector{Float64}, b::Int64,
		fs0m::SensitiveFloat, fs1m::SensitiveFloat, 
		E_F::SensitiveFloat, var_F::SensitiveFloat)

	clear!(fs0m)
	for star_mc in star_mcs[:, s]
		accum_star!(star_mc, m_pos, vs.gamma[b], fs0m)
	end

	clear!(fs1m)
	for i = 1:2
		theta_dir = (i == 1) ? 1. : -1.
		theta_i = (i == 1) ? vs.theta : 1. - vs.theta

		for j in 1:[6,8][i]
			for k = 1:3
				accum_galaxy!(gal_mcs[k, j, i, s], m_pos, vs.zeta[b],
					theta_i, theta_dir, vs.Xi, galaxy_prototypes[i][j].sigmaTilde, fs1m)
			end
		end
	end
	
	E_F.v += (1. - vs.chi) * fs0m.v + vs.chi * fs1m.v

	E_F.d[ids.chi, s] += fs1m.v - fs0m.v
	for i in 1:length(fs0m.index)
		p = fs0m.index[i]
		E_F.d[p, s] += (1. - vs.chi) * fs0m.d[i]
	end
	for i in 1:length(fs1m.index)
		p = fs1m.index[i]
		E_F.d[p, s] += vs.chi * fs1m.d[i]
	end

	diff10 = fs1m.v - fs0m.v
	diff10_sq = diff10^2
	chi_var = (vs.chi * (1. - vs.chi)) # cache these?
	chi_var_d = 1. - 2 * vs.chi
	var_F.v += chi_var * diff10_sq
	var_F.d[ids.chi, s] += chi_var_d * diff10_sq
	for i in 1:length(fs0m.index)
		p = fs0m.index[i]
		var_F.d[p, s] -= chi_var * 2 * diff10 * fs0m.d[i]
	end
	for i in 1:length(fs1m.index)
		p = fs1m.index[i]
		var_F.d[p, s] += chi_var * 2 * diff10 * fs1m.d[i]
	end
end


function accum_pixel_ret!(tile_sources::Vector{Int64}, x_nbm::Float64,
		E_F::SensitiveFloat, var_F::SensitiveFloat, ret::SensitiveFloat)
	ret.v += x_nbm * (log(E_F.v) - var_F.v / (2. * E_F.v^2))
	ret.v -= E_F.v

	for s in tile_sources, p in 1:size(E_F.d, 1)
		ret.d[p, s] += x_nbm * (E_F.d[p, s] / E_F.v
			- 0.5 * (E_F.v^2 * var_F.d[p, s] - 
				var_F.v * 2 * E_F.v * E_F.d[p, s]) 
					./  E_F.v^4)
		ret.d[p, s] -= E_F.d[p, s]
	end
end


function tile_range(tile::ImageTile, tile_width::Int64)
	h1 = 1 + (tile.hh - 1) * tile_width 
	h2 = min(tile.hh * tile_width, tile.img.H)
	w1 = 1 + (tile.ww - 1) * tile_width 
	w2 = min(tile.ww * tile_width, tile.img.W)
	h1:h2, w1:w2
end


function local_sources(tile::ImageTile, mp::ModelParams)
	local_subset = Array(Int64, 0)

	tr = mp.tile_width / 2.  # tile radius
	tc1 = tr + (tile.hh - 1) * mp.tile_width
	tc2 = tr + (tile.ww - 1) * mp.tile_width

	for s in 1:mp.S
		pc = mp.patches[s].center  # patch center
		pr = mp.patches[s].radius  # patch radius

		if abs(pc[1] - tc1) <= (pr + tr) && abs(pc[2] - tc2) <= (pr + tr)
			push!(local_subset, s)
		end
	end

	local_subset
end


function elbo_likelihood!(tile::ImageTile, mp::ModelParams, 
		star_mcs::Array{BvnComponent, 2}, gal_mcs::Array{BvnComponent, 4}, 
		accum::SensitiveFloat)
	tile_sources = local_sources(tile, mp)
	h_range, w_range = tile_range(tile, mp.tile_width)

	if length(tile_sources) == 0  # special case---for speed
		num_pixels = length(h_range) * length(w_range)
		ep = tile.img.epsilon
		tile_x = sum(tile.img.pixels[h_range, w_range])
		accum.v += tile_x * log(ep) - num_pixels * ep
		return
	end

	# TODO: revise for new model!!!
	const star_b_index = [2, 3, 3 + tile.img.b]
	const gal_b_index = [2, 3, 9 + tile.img.b, 15, 16, 17, 18]

	fs0m = zero_sensitive_float(star_b_index)
	fs1m = zero_sensitive_float(gal_b_index)

    E_F = zero_sensitive_float(tile_sources, all_params)
    var_F = zero_sensitive_float(tile_sources, all_params)

	for w in w_range, h in h_range
		clear!(E_F)  #serious bottleneck
		E_F.v = tile.img.epsilon
		clear!(var_F)

		m_pos = Float64[h - 0.5, w - 0.5]
		for s in tile_sources
			accum_pixel_source_stats!(star_mcs, gal_mcs, mp.vp[s],
					s, m_pos, tile.img.b, fs0m, fs1m, E_F, var_F)
		end

		accum_pixel_ret!(tile_sources, tile.img.pixels[h, w], E_F, var_F, accum)
	end
end


function elbo_likelihood!(img::Image, mp::ModelParams, accum::SensitiveFloat)
	accum.v += -sum(lfact(img.pixels))

	star_mcs, gal_mcs = load_bvn_mixtures(img.psf, mp)

	WW = int(ceil(img.W / mp.tile_width))
	HH = int(ceil(img.H / mp.tile_width))
	for ww in 1:WW, hh in 1:HH
		tile = ImageTile(hh, ww, img)
		# might get a speedup from subsetting the mp here
		elbo_likelihood!(tile, mp, star_mcs, gal_mcs, accum)
	end
end


function elbo_likelihood(blob::Blob, mp::ModelParams)
	ret = zero_sensitive_float([1:mp.S], all_params)
	for img in blob
		elbo_likelihood!(img, mp, ret)
	end
	ret
end


function subtract_kl_c!(d::Int64, i::Int64, s::Int64, mp::ModelParams, 
		accum::SensitiveFloat)
	vs = mp.vp[s]
    beta, lambda = (vs[ids.beta[:, i]], vs[ids.lambda[:, i]])
    Omega, Lambda = (mp.pp.Omega[i][:, d], mp.pp.Lambda[i][d])

	diff = Omega - beta
	Lambda_inv = Lambda^-1  # cache this!
	half_kappa = .5 * vs[ids.kappa[d, i]]

	ret = sum(diag(Lambda_inv) .* lambda)
	ret += (diff' * Lambda_inv * diff)[]
	ret += -4 - sum(log(lambda)) + logdet(Lambda)
	accum.v -= ret * half_kappa

	accum.d[ids.kappa[d, i], s] -= .5 * ret
	accum.d[ids.beta[:, i], s] -= half_kappa * 2Lambda_inv * -diff
	accum.d[ids.lambda[:, i], s] -= half_kappa * diag(Lambda_inv)
	accum.d[ids.lambda[:, i], s] -= half_kappa ./ -lambda
end


function subtract_kl_k!(i::Int64, s::Int64, mp::ModelParams, accum::SensitiveFloat)
	kappa = mp.vp[s][ids.kappa[:, i]]
	for d in 1:length(mp.pp.Psi[i])
		log_ratio = log(kappa[d] / mp.pp.Psi[i][d])
		accum.v -= kappa[d] * log_ratio
		accum.d[ids.kappa[d, i] , s] -= 1 + log_ratio
	end
end


function subtract_kl_r!(i::Int64, s::Int64, mp::ModelParams, accum::SensitiveFloat)
	vs = mp.vp[s]
	gamma_si = mp.vp[s][ids.gamma[i]]
	zeta_si = mp.vp[s][ids.zeta[i]]

	digamma_gamma = digamma(gamma_si)
	zeta_Phi_ratio = (zeta_si - mp.pp.Phi[i]) / mp.pp.Phi[i]
	shape_diff = gamma_si - mp.pp.Upsilon[i]

	accum.v -= shape_diff * digamma_gamma
	accum.v -= -lgamma(gamma_si) + lgamma(mp.pp.Upsilon[i])
	accum.v -= mp.pp.Upsilon[i] * (log(mp.pp.Phi[i]) - log(zeta_si))
	accum.v -= gamma_si * zeta_Phi_ratio

	accum.d[ids.gamma[i], s] -= shape_diff * polygamma(1, gamma_si)
	accum.d[ids.gamma[i], s] -= zeta_Phi_ratio

	accum.d[ids.zeta[i], s] -= -mp.pp.Upsilon[i] / zeta_si
	accum.d[ids.zeta[i], s] -= gamma_si / mp.pp.Phi[i]
end


function subtract_kl_a!(s::Int64, mp::ModelParams, accum::SensitiveFloat)
	chi_s = mp.vp[s][ids.chi]
	Delta = mp.pp.Delta

	accum.v -= chi_s * (log(chi_s) - log(Delta))
	accum.v -= (1. - chi_s) * (log(1. - chi_s) - log(1. - Delta))

	accum.d[ids.chi, s] -= (log(chi_s) - log(Delta)) + 1
	accum.d[ids.chi, s] -= -(log(1. - chi_s) - log(1. - Delta)) - 1.
end


function subtract_kl!(mp::ModelParams, accum::SensitiveFloat)
	for s in 1:M.S
		subtract_kl_a!(s, mp, accum)
		for i in 1:2
			subtract_kl_r!(i, s, mp, accum)
			subtract_kl_k!(i, s, mp, accum)
			subtract_kl_c!(i, s, mp, accum)
		end
	end
end


function elbo(blob::Blob, mp::ModelParams)
	ret = elbo_likelihood(blob, mp)
	subtract_kl!(mp, ret)
	ret
end


end

