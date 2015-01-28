#!/usr/bin/env julia

using Celeste
using CelesteTypes
using Base.Test

using Distributions
import GSL.deriv_central

import Synthetic

const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")


# verify derivatives of fun_to_test by finite differences
function test_by_finite_differences(fun_to_test::Function, mp::ModelParams)
	f::SensitiveFloat = fun_to_test(mp)

	for s in 1:mp.S
		for p1 in 1:length(f.param_index)
			p0 = f.param_index[p1]

			fun_to_test_2(epsilon::Float64) = begin
				vp_local = deepcopy(mp.vp)
				vp_local[s][p0] += epsilon
				mp_local = ModelParams(vp_local, mp.pp, mp.patches, mp.tile_width)
				f_local::SensitiveFloat = fun_to_test(mp_local)
				f_local.v
			end

			numeric_deriv, abs_err = deriv_central(fun_to_test_2, 0., 1e-3)
#			println("deriv #$p0 (s: $s): $numeric_deriv vs $(f.d[p1, s]) [tol: $abs_err]")
            obs_err = abs(numeric_deriv - f.d[p1, s]) 
			@test obs_err < 1e-11 || abs_err < 1e-4 || abs_err / abs(numeric_deriv) < 1e-5
			@test_approx_eq_eps numeric_deriv f.d[p1, s] 10abs_err
		end
	end
end


function perturb_params(mp) # for testing derivatives != 0
	for vs in mp.vp
		vs[ids.chi] = 0.6
		vs[ids.mu[1]] += .8
		vs[ids.mu[2]] -= .7
		vs[ids.gamma] /= 10
		vs[ids.zeta] *= 25.
		vs[ids.theta] -= 0.05
		vs[ids.rho] += 0.05
		vs[ids.phi] += pi/10
		vs[ids.sigma] *= 0.8
		vs[ids.beta] += 0.5
		vs[ids.lambda] =  1e-1
	end
end


#########################

const star_fluxes = [4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]
const galaxy_fluxes = [1.377666E+01, 5.635334E+01, 1.258656E+02, 
					1.884264E+02, 2.351820E+02] * 100  # 1x wasn't bright enough

function standard_ce(pos, is_star::Bool)
	CatalogEntry(pos, is_star, star_fluxes, galaxy_fluxes, 0.1, .7, pi/4, 4.)
end

function gen_one_star_dataset(; perturb=true)
	srand(1)
	blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
	for b in 1:5
		blob0[b].H, blob0[b].W = 20, 23
	end
	one_body = [standard_ce([10.1, 12.2], true),]
   	blob = Synthetic.gen_blob(blob0, one_body)
	mp = ModelInit.cat_init(one_body)
    if perturb
        perturb_params(mp)
    end

	blob, mp, one_body
end

function gen_one_galaxy_dataset(; perturb=true)
	srand(1)
	blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
	for b in 1:5
		blob0[b].H, blob0[b].W = 20, 23
	end
	one_body = [standard_ce([8.5, 9.6], false),]
   	blob = Synthetic.gen_blob(blob0, one_body)
	mp = ModelInit.cat_init(one_body)
    if perturb
        perturb_params(mp)
    end

	blob, mp, one_body
end

function gen_three_body_model(; perturb=true)
	srand(1)
	blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
	for b in 1:5
		blob0[b].H, blob0[b].W = 112, 238
	end
	three_bodies = [
		standard_ce([4.5, 3.6], false),
		standard_ce([60.1, 82.2], true),
		standard_ce([71.3, 100.4], false),
	]
   	blob = Synthetic.gen_blob(blob0, three_bodies)
	mp = ModelInit.cat_init(three_bodies)
    if perturb
        perturb_params(mp)
    end

	blob, mp, three_bodies
end

#########################

function test_brightness_derivs()
	blob, mp0, three_bodies = gen_three_body_model()

	for i = 1:2
		for b = [3,4,2,5,1]
			function wrap_source_brightness(mp)
				sb = ElboDeriv.SourceBrightness(mp.vp[1])
				ret = zero_sensitive_float([1,2,3], all_params)
				ret.v = sb.E_l_a[b, i].v
				ret.d[:, 1] = sb.E_l_a[b, i].d
				ret
			end
			test_by_finite_differences(wrap_source_brightness, mp0)

			function wrap_source_brightness_3(mp)
				sb = ElboDeriv.SourceBrightness(mp.vp[1])
				ret = zero_sensitive_float([1,2,3], all_params)
				ret.v = sb.E_ll_a[b, i].v
				ret.d[:, 1] = sb.E_ll_a[b, i].d
				ret
			end
			test_by_finite_differences(wrap_source_brightness_3, mp0)
		end
	end
end


function test_accum_pos()
	blob, mp, body = gen_one_galaxy_dataset()

	function wrap_star(mmp)
		star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[3].psf, mmp)
		fs0m = zero_sensitive_float([1], star_pos_params)
		ElboDeriv.accum_star_pos!(star_mcs[1,1], [9, 10.], fs0m)
		fs0m
	end
	test_by_finite_differences(wrap_star, mp)

	function wrap_galaxy(mmp)
		star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[3].psf, mmp)
		fs1m = zero_sensitive_float([1], galaxy_pos_params)
		ElboDeriv.accum_galaxy_pos!(gal_mcs[1,1,1,1], [9, 10.], fs1m)
		fs1m
	end
	test_by_finite_differences(wrap_galaxy, mp)
end


function test_accum_pixel_source_derivs()
	blob, mp0, body = gen_one_galaxy_dataset()

	function wrap_apss_ef(mmp)
		star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[1].psf, mmp)
		fs0m = zero_sensitive_float([1], star_pos_params)
		fs1m = zero_sensitive_float([1], galaxy_pos_params)
		E_G = zero_sensitive_float([1], all_params)
		var_G = zero_sensitive_float([1], all_params)
		sb = ElboDeriv.SourceBrightness(mmp.vp[1])
		m_pos = [9, 10.]
		ElboDeriv.accum_pixel_source_stats!(sb, star_mcs, gal_mcs,
			mmp.vp[1], 1, 1, m_pos, 1, fs0m, fs1m, E_G, var_G)
		E_G
	end
	test_by_finite_differences(wrap_apss_ef, mp0)

	function wrap_apss_varf(mmp)
		star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[3].psf, mmp)
		fs0m = zero_sensitive_float([1], star_pos_params)
		fs1m = zero_sensitive_float([1], galaxy_pos_params)
		E_G = zero_sensitive_float([1], all_params)
		var_G = zero_sensitive_float([1], all_params)
		sb = ElboDeriv.SourceBrightness(mmp.vp[1])
		m_pos = [9, 10.]
		ElboDeriv.accum_pixel_source_stats!(sb, star_mcs, gal_mcs,
			mmp.vp[1], 1, 1, m_pos, 3, fs0m, fs1m, E_G, var_G)
		var_G
	end
	test_by_finite_differences(wrap_apss_varf, mp0)

end

function test_elbo_derivs()
	blob, mp0, body = gen_one_galaxy_dataset()

	function wrap_likelihood_b1(mmp)
		ElboDeriv.elbo_likelihood([blob[1]], mmp)
	end
	test_by_finite_differences(wrap_likelihood_b1, mp0)

	function wrap_likelihood_b5(mmp)
		ElboDeriv.elbo_likelihood([blob[5]], mmp)
	end
	test_by_finite_differences(wrap_likelihood_b5, mp0)

	function wrap_elbo(mmp)
		ElboDeriv.elbo([blob], mmp)
	end
	test_by_finite_differences(wrap_elbo, mp0)
end


function test_kl_divergence_values()
	blob, mp, three_bodies = gen_three_body_model()

	s = 1
	i = 1
	d = 1
	sample_size = 2_000_000

	function test_kl(q_dist, p_dist, subtract_kl_fun!)
		q_samples = rand(q_dist, sample_size)
		empirical_kl_samples = logpdf(q_dist, q_samples) - logpdf(p_dist, q_samples)
		empirical_kl = mean(empirical_kl_samples)
		accum = zero_sensitive_float([s], all_params)
		subtract_kl_fun!(accum)
		exact_kl = -accum.v
		tol = 4 * std(empirical_kl_samples) / sqrt(sample_size)
		min_diff = 1e-2 * std(empirical_kl_samples) / sqrt(sample_size)
		@test_approx_eq_eps empirical_kl exact_kl tol
	end

	vs = mp.vp[s]

	# a
	q_a = Bernoulli(vs[ids.chi])
	p_a = Bernoulli(mp.pp.Delta)
	test_kl(q_a, p_a, (accum) -> ElboDeriv.subtract_kl_a!(s, mp, accum))

	# r
	q_r = Gamma(vs[ids.gamma[i]], vs[ids.zeta[i]])
	p_r = Gamma(mp.pp.Upsilon[i], mp.pp.Phi[i])
	test_kl(q_r, p_r, (accum) -> ElboDeriv.subtract_kl_r!(i, s, mp, accum))

	# k
	q_k = Categorical(vs[ids.kappa[:, i]])
	p_k = Categorical(mp.pp.Psi[i])
	test_kl(q_k, p_k, (accum) -> ElboDeriv.subtract_kl_k!(i, s, mp, accum))

	# c
	mp.pp.Omega[i][:, d] = vs[ids.beta[:, i]]
	mp.pp.Lambda[i][d] = diagm(vs[ids.lambda[:, i]])
	q_c = MvNormal(vs[ids.beta[:, i]], diagm(vs[ids.lambda[:, i]]))
	p_c = MvNormal(mp.pp.Omega[i][:, d], mp.pp.Lambda[i][d])
	function sklc(accum)
		ElboDeriv.subtract_kl_c!(d, i, s, mp, accum)
		accum.v /= vs[ids.kappa[d, i]]
	end
	test_kl(q_c, p_c, sklc)
end


function test_kl_divergence_derivs()
	blob, mp0, three_bodies = gen_three_body_model()

	function wrap_kl_a(mp)
		accum = zero_sensitive_float([1:3], all_params)
		ElboDeriv.subtract_kl_a!(1, mp, accum)
		accum
	end
	test_by_finite_differences(wrap_kl_a, mp0)

	function wrap_kl_r(mp)
		accum = zero_sensitive_float([1:3], all_params)
		ElboDeriv.subtract_kl_r!(1, 1, mp, accum)
		accum
	end
	test_by_finite_differences(wrap_kl_r, mp0)

	function wrap_kl_k(mp)
		accum = zero_sensitive_float([1:3], all_params)
		ElboDeriv.subtract_kl_k!(1, 1, mp, accum)
		accum
	end
	test_by_finite_differences(wrap_kl_k, mp0)

	function wrap_kl_c(mp)
		accum = zero_sensitive_float([1:3], all_params)
		ElboDeriv.subtract_kl_c!(1, 1, 1, mp, accum)
		accum
	end
	test_by_finite_differences(wrap_kl_c, mp0)
end

#########################


function true_star_init()
	blob, mp, body = gen_one_star_dataset()

	flx = body[1].star_fluxes
	colors = log(flx[2:5] ./ flx[1:4])

	mp.vp[1][ids.mu] = body[1].pos
	mp.vp[1][ids.chi] = 1e-3
	mp.vp[1][ids.zeta] = 1e-4
	mp.vp[1][ids.gamma] = flx[3] ./ mp.vp[1][ids.zeta]
	mp.vp[1][ids.lambda] = 1e-4
	mp.vp[1][ids.beta[:, 1]] = mp.vp[1][ids.beta[:, 2]] = colors

	blob, mp, body
end


function test_that_variance_is_low()
	# very peaked variational distribution---variance for F(m) should be low
	blob, mp, body = true_star_init()

	star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[3].psf, mp)
	fs0m = zero_sensitive_float([1], star_pos_params)
	fs1m = zero_sensitive_float([1], galaxy_pos_params)
	E_G = zero_sensitive_float([1], all_params)
	var_G = zero_sensitive_float([1], all_params)
	sb = ElboDeriv.SourceBrightness(mp.vp[1])
	m_pos = [10, 12.]
	ElboDeriv.accum_pixel_source_stats!(sb, star_mcs, gal_mcs,
		mp.vp[1], 1, 1, m_pos, 3, fs0m, fs1m, E_G, var_G)

	@test 0 < var_G.v < 1e-2 * E_G.v^2
end


function test_that_star_truth_is_more_likely()
	blob, mp, body = true_star_init()
	best = ElboDeriv.elbo_likelihood(blob, mp)

	for bad_chi in [.3, .5, .9]
		mp_chi = deepcopy(mp)
		mp_chi.vp[1][ids.chi] = bad_chi
		bad_chi = ElboDeriv.elbo_likelihood(blob, mp_chi)
		@test best.v > bad_chi.v
	end

	for h2 in -2:2
		for w2 in -2:2
			if !(h2 == 0 && w2 == 0)
				mp_mu = deepcopy(mp)
				mp_mu.vp[1][ids.mu] += [h2 * .5, w2 * .5]
				bad_mu = ElboDeriv.elbo_likelihood(blob, mp_mu)
				@test best.v > bad_mu.v
			end
		end
	end

	for delta in [.7, .9, 1.1, 1.3]
		mp_gamma = deepcopy(mp)
		mp_gamma.vp[1][ids.gamma] *= delta
		bad_gamma = ElboDeriv.elbo_likelihood(blob, mp_gamma)
		@test best.v > bad_gamma.v
	end

	for b in 1:4
		for delta in [.7, .9, 1.1, 1.3]
			mp_beta = deepcopy(mp)
			mp_beta.vp[1][ids.beta[b], 1] *= delta
			bad_beta = ElboDeriv.elbo_likelihood(blob, mp_beta)
			@test best.v > bad_beta.v
		end
	end
end


function test_that_galaxy_truth_is_more_likely()
	blob, mp, body = gen_one_galaxy_dataset(perturb=false)
	best = ElboDeriv.elbo_likelihood(blob, mp)

	for bad_chi in [.3, .5, .9]
		mp_chi = deepcopy(mp)
		mp_chi.vp[1][ids.chi] = bad_chi
		bad_chi = ElboDeriv.elbo_likelihood(blob, mp_chi)
		@test best.v > bad_chi.v
	end

	for h2 in -2:2
		for w2 in -2:2
			if !(h2 == 0 && w2 == 0)
				mp_mu = deepcopy(mp)
				mp_mu.vp[1][ids.mu] += [h2 * .5, w2 * .5]
				bad_mu = ElboDeriv.elbo_likelihood(blob, mp_mu)
				@test best.v > bad_mu.v
			end
		end
	end

    for bad_scale in [.8, 1.2]
		mp_gamma = deepcopy(mp)
		mp_gamma.vp[1][ids.gamma] *= bad_scale^2
		mp_gamma.vp[1][ids.zeta] /= bad_scale  # keep variance the same
		bad_gamma = ElboDeriv.elbo_likelihood(blob, mp_gamma)
		@test best.v > bad_gamma.v
	end

    for n in [:rho, :phi, :sigma]
        for bad_scale in [.8, 1.2]
            mp_bad = deepcopy(mp)
            mp_bad.vp[1][ids.(n)] *= bad_scale
            bad_elbo = ElboDeriv.elbo_likelihood(blob, mp_bad)
            @test best.v > bad_elbo.v
        end
    end

	for b in 1:4
		for delta in [-.3, .3]
			mp_beta = deepcopy(mp)
			mp_beta.vp[1][ids.beta[b, 2]] += delta
			bad_beta = ElboDeriv.elbo_likelihood(blob, mp_beta)
			@test best.v > bad_beta.v
		end
	end
end


function test_star_optimization()
	blob, mp, body = true_star_init()

	flx = body[1].star_fluxes
	true_colors = log(flx[2:5] ./ flx[1:4])
	
	OptimizeElbo.maximize_likelihood(blob, mp)

	@test_approx_eq mp.vp[1][ids.chi] 0.0001
	@test_approx_eq_eps mp.vp[1][ids.mu[1]] 10.1 0.1
	@test_approx_eq_eps mp.vp[1][ids.mu[2]] 12.2 0.1
	@test_approx_eq_eps (mp.vp[1][ids.gamma[1]] * mp.vp[1][ids.zeta[1]]) flx[3] 1e2
	for b in 1:4
		@test_approx_eq_eps mp.vp[1][ids.beta[b, 1]] true_colors[b] 0.1
	end
	
end


function test_galaxy_optimization()
	blob, mp, body = gen_one_galaxy_dataset()

	flx = body[1].gal_fluxes
	colors = log(flx[2:5] ./ flx[1:4])
	
	OptimizeElbo.maximize_likelihood(blob, mp)

	@test_approx_eq mp.vp[1][ids.chi] 0.9999
	@test_approx_eq_eps mp.vp[1][ids.mu[1]] 8.5 0.1
	@test_approx_eq_eps mp.vp[1][ids.mu[2]] 9.6 0.1
	@test_approx_eq_eps mp.vp[1][ids.rho] .7 0.05
	@test_approx_eq_eps mp.vp[1][ids.phi] pi/4 pi/20
	@test_approx_eq_eps mp.vp[1][ids.sigma] 4. 0.2
	@test_approx_eq_eps (mp.vp[1][ids.gamma[2]] * mp.vp[1][ids.zeta[2]]) flx[3] 1e2
	for b in 1:4
		@test_approx_eq_eps mp.vp[1][ids.beta[b, 2]] colors[b] 0.1
	end
end


function test_peak_init_galaxy_optimization()
	blob, mp, body = gen_one_galaxy_dataset()
	mp = ModelInit.peak_init(blob)

	flx = body[1].gal_fluxes
	colors = log(flx[2:5] ./ flx[1:4])

	OptimizeElbo.maximize_likelihood(blob, mp)

	@test_approx_eq mp.vp[1][ids.chi] 0.9999
	@test_approx_eq_eps mp.vp[1][ids.mu[1]] 8.5 0.1
	@test_approx_eq_eps mp.vp[1][ids.mu[2]] 9.6 0.1
	@test_approx_eq_eps mp.vp[1][ids.rho] .7 0.05
	@test_approx_eq_eps mp.vp[1][ids.phi] pi/4 pi/20
	@test_approx_eq_eps mp.vp[1][ids.sigma] 4. 0.2
	@test_approx_eq_eps (mp.vp[1][ids.gamma[2]] * mp.vp[1][ids.zeta[2]]) flx[3] 1e2
	for b in 1:4
		@test_approx_eq_eps mp.vp[1][ids.beta[b, 2]] colors[b] 0.1
	end
end


function test_peak_init_2body_optimization()
	srand(1)
	blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")

	two_bodies = [
		standard_ce([11.1, 21.2], true),
		standard_ce([15.3, 31.4], false),
	]

   	blob = Synthetic.gen_blob(blob0, two_bodies)
	mp = ModelInit.peak_init(blob) #one giant tile, giant patches
	@test mp.S == 2

	OptimizeElbo.maximize_likelihood(blob, mp)

	@test_approx_eq mp.vp[1][ids.chi] 0.0001
	@test_approx_eq mp.vp[2][ids.chi] 0.9999
	@test_approx_eq_eps mp.vp[2][ids.theta] 0.1 0.08
	@test_approx_eq_eps mp.vp[1][ids.mu[1]] 11.1 0.1
	@test_approx_eq_eps mp.vp[1][ids.mu[2]] 21.2 0.1
	@test_approx_eq_eps mp.vp[2][ids.mu[1]] 15.3 0.1
	@test_approx_eq_eps mp.vp[2][ids.mu[2]] 31.4 0.1
	@test_approx_eq_eps mp.vp[1][ids.rho] .7 0.05
	@test_approx_eq_eps mp.vp[1][ids.phi] pi/4 pi/20
	@test_approx_eq_eps mp.vp[1][ids.sigma] 4. 0.2
end


function test_local_sources()
	srand(1)
	blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
	for b in 1:5
		blob0[b].H, blob0[b].W = 112, 238
	end

	three_bodies = [
		standard_ce([4.5, 3.6], false),
		standard_ce([60.1, 82.2], true),
		standard_ce([71.3, 100.4], false),
	]

   	blob = Synthetic.gen_blob(blob0, three_bodies)

	mp = ModelInit.cat_init(three_bodies, patch_radius=20., tile_width=1000)
	@test mp.S == 3

	tile = ImageTile(1, 1, blob[3])
	subset1000 = ElboDeriv.local_sources(tile, mp)
	@test subset1000 == [1,2,3]

	mp.tile_width=10

	subset10 = ElboDeriv.local_sources(tile, mp)
	@test subset10 == [1]

	last_tile = ImageTile(11, 24, blob[3])
	last_subset = ElboDeriv.local_sources(last_tile, mp)
	@test length(last_subset) == 0

	pop_tile = ImageTile(7, 9, blob[3])
	pop_subset = ElboDeriv.local_sources(pop_tile, mp)
	@test pop_subset == [2,3]
end


function test_local_sources_2()
	srand(1)
	blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
	one_body = [standard_ce([50., 50.], true),]

   	for b in 1:5 blob0[b].H, blob0[b].W = 100, 100 end
	small_blob = Synthetic.gen_blob(blob0, one_body)

   	for b in 1:5 blob0[b].H, blob0[b].W = 400, 400 end
	big_blob = Synthetic.gen_blob(blob0, one_body)

	mp = ModelInit.cat_init(one_body, patch_radius=35., tile_width=2)

	qx = 0
	for ww=1:50,hh=1:50
		tile = ImageTile(hh, ww, small_blob[2])
		if length(ElboDeriv.local_sources(tile, mp)) > 0
			qx += 1
		end
	end

	@test qx == (36 * 2)^2 / 4

	qy = 0
	for ww=1:200,hh=1:200
		tile = ImageTile(hh, ww, big_blob[1])
		if length(ElboDeriv.local_sources(tile, mp)) > 0
			qy += 1
		end
	end

	@test qy == qx
end


function test_tiling()
	srand(1)
	blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
	for b in 1:5
		blob0[b].H, blob0[b].W = 112, 238
	end
	three_bodies = [
#		standard_ce([4.5, 3.6], false, star_fluxes, galaxy_fluxes / 50, 0.1, [3, 0., 3.]),
		standard_ce([60.1, 82.2], true, star_fluxes / 50, galaxy_fluxes, 0.1, [3, 0., 3.]),
#		standard_ce([71.3, 100.4], false, star_fluxes, galaxy_fluxes / 50, 0.1, [3, 0., 3.]),
	]
   	blob = Synthetic.gen_blob(blob0, three_bodies)
	mp = ModelInit.cat_init(three_bodies)

	println(median(blob[3].pixels), "  ", blob[3].epsilon * blob[3].iota)

	elbo = ElboDeriv.elbo(blob, mp)

	mp2 = ModelInit.cat_init(three_bodies, tile_width=10)
	elbo_tiles = ElboDeriv.elbo(blob, mp2)
	@test_approx_eq_eps elbo_tiles.v elbo.v 1e-5

	mp3 = ModelInit.cat_init(three_bodies, patch_radius=30.)
	elbo_patches = ElboDeriv.elbo(blob, mp3)
	@test_approx_eq_eps elbo_patches.v elbo.v 1e-5

	for s in 1:mp.S
		for i in 1:length(all_params)
			@test_approx_eq_eps elbo_tiles.d[i, s] elbo.d[i, s] 1e-5
			@test_approx_eq_eps elbo_patches.d[i, s] elbo.d[i, s] 1e-5
		end
	end

	mp4 = ModelInit.cat_init(three_bodies, patch_radius=35., tile_width=10)
	elbo_both = ElboDeriv.elbo(blob, mp4)
	@test_approx_eq_eps elbo_both.v elbo.v 1e-1

	for s in 1:mp.S
		for i in 1:length(all_params)
			@test_approx_eq_eps elbo_both.d[i, s] elbo.d[i, s] 1e-1
			println(abs(elbo_both.d[i, s] - elbo.d[i, s]))
		end
	end
end


function test_sky_noise_estimates()
	blobs = Array(Blob, 2)
	blobs[1], mp, three_bodies = gen_three_body_model()  # synthetic
	blobs[2] = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")  # real

	for blob in blobs
		for b in 1:5
			sdss_sky_estimate = blob[b].epsilon * blob[b].iota
			crude_estimate = median(blob[b].pixels)
			@test_approx_eq_eps sdss_sky_estimate / crude_estimate 1. .3
		end
	end
end

function test_full_elbo_optimization()
	blob, mp, body = gen_one_galaxy_dataset(perturb=true)

	flx = body[1].gal_fluxes
	colors = log(flx[2:5] ./ flx[1:4])
	
	OptimizeElbo.maximize_elbo(blob, mp)

	@test_approx_eq mp.vp[1][ids.chi] 0.9999
	@test_approx_eq_eps mp.vp[1][ids.mu[1]] 8.5 0.1
	@test_approx_eq_eps mp.vp[1][ids.mu[2]] 9.6 0.1
	@test_approx_eq_eps mp.vp[1][ids.rho] .7 0.05
	@test_approx_eq_eps mp.vp[1][ids.phi] pi/4 pi/20
	@test_approx_eq_eps mp.vp[1][ids.sigma] 4. 0.2
	@test_approx_eq_eps (mp.vp[1][ids.gamma[2]] * mp.vp[1][ids.zeta[2]]) flx[3] 1e2
	for b in 1:4
		@test_approx_eq_eps mp.vp[1][ids.beta[b, 2]] colors[b] 0.1
	end
end


function test_coordinates_vp_conversion()
	blob, mp, three_bodies = gen_three_body_model()

	xs = OptimizeElbo.vp_to_coordinates(deepcopy(mp.vp), [ids.lambda[:]])
	vp_new = deepcopy(mp.vp)
	OptimizeElbo.coordinates_to_vp!(deepcopy(xs), vp_new, [ids.lambda[:]])

	@test length(xs) + 3 * 2 * (4 + 1) == 
			length(vp_new[1]) * length(vp_new) == 
			length(mp.vp[1]) * length(mp.vp)

	for s in 1:3
		for p in all_params
			@test_approx_eq mp.vp[s][p] vp_new[s][p]
		end
	end
end


####################################################

function test_coadd_cat_init_is_more_likely()  # on a real stamp
	stamp_id = "5.0562-0.0643"
	blob = SDSS.load_stamp_blob(dat_dir, stamp_id)
	cat_entries = SDSS.load_stamp_catalog(dat_dir, "s82-$stamp_id", blob)
	mp = ModelInit.cat_init(cat_entries)
	best = ElboDeriv.elbo_likelihood(blob, mp)

	# s is the brightest source: a galaxy!
	s = 1
	println(cat_entries[s])

	for bad_scale in [.8, 1.2]
		mp_gamma = deepcopy(mp)
		mp_gamma.vp[s][ids.gamma] *= bad_scale^2
		mp_gamma.vp[s][ids.zeta] /= bad_scale  # keep variance the same
		bad_gamma = ElboDeriv.elbo_likelihood(blob, mp_gamma)
		@test best.v > bad_gamma.v
	end

    for n in [:rho, :phi, :sigma]
        for bad_scale in [.8, 1.2]
            mp_bad = deepcopy(mp)
            mp_bad.vp[s][ids.(n)] *= bad_scale
            bad_elbo = ElboDeriv.elbo_likelihood(blob, mp_bad)
            @test best.v > bad_elbo.v
        end
    end

	for bad_chi in [.3, .7]
		mp_chi = deepcopy(mp)
		mp_chi.vp[s][ids.chi] = bad_chi
		bad_chi = ElboDeriv.elbo_likelihood(blob, mp_chi)
		@test best.v > bad_chi.v
	end

	for h2 in -2:2
		for w2 in -2:2
			if !(h2 == 0 && w2 == 0)
				mp_mu = deepcopy(mp)
				mp_mu.vp[s][ids.mu] += [0.5h2, 0.5w2]
				bad_mu = ElboDeriv.elbo_likelihood(blob, mp_mu)
				@test best.v > bad_mu.v
			end
		end
	end

	for b in 1:4
		for delta in [-.3, .3]
			mp_beta = deepcopy(mp)
			mp_beta.vp[s][ids.beta[b, :]] += delta
			bad_beta = ElboDeriv.elbo_likelihood(blob, mp_beta)
			@test best.v > bad_beta.v
		end
	end
end


function test_real_stamp_optimization()
	blob = SDSS.load_stamp_blob(dat_dir, "5.0562-0.0643")
	cat_entries = SDSS.load_stamp_catalog(dat_dir, "s82-5.0562-0.0643", blob)

	for ce in cat_entries
		println(ce)
	end

	mp = ModelInit.cat_init(cat_entries)
	OptimizeElbo.maximize_elbo(blob, mp)
end


function test_tiny_image_tiling()
	blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
	pc = PsfComponent(1./3, zeros(2), 1e-4 * eye(2))
	trivial_psf = [pc, pc, pc]
	pixels = ones(100, 1) * 12
	pixels[98:100, 1] = [1e3, 1e4, 1e5]
	img = Image(3, 1, pixels, 3, blob0[3].wcs, 3., 4, trivial_psf, 1, 1, 1)
	catalog = [standard_ce([100., 1], true),]
	catalog[1].star_fluxes = ones(5) * 1e5

	mp0 = ModelInit.cat_init(catalog)
	accum0 = zero_sensitive_float([1], all_params)
	ElboDeriv.elbo_likelihood!(img, mp0, accum0)

	mp_tiles = ModelInit.cat_init(catalog, patch_radius=10., tile_width=2)
	accum_tiles = zero_sensitive_float([1], all_params)
	ElboDeriv.elbo_likelihood!(img, mp_tiles, accum_tiles)

	mp_tiles2 = ModelInit.cat_init(catalog, patch_radius=10., tile_width=5)
	accum_tiles2 = zero_sensitive_float([1], all_params)
	ElboDeriv.elbo_likelihood!(img, mp_tiles, accum_tiles2)
	@test_approx_eq accum_tiles.v accum_tiles2.v

	@test_approx_eq_eps accum0.v accum_tiles.v 100.
end


function test_util_bvn_cov()
    rho = .7
    phi = pi/5
    sigma = 2.

    manual_11 = sigma^2 * (1 + (rho^2 - 1) * (sin(phi))^2)
    util_11 = Util.get_bvn_cov(rho, phi, sigma)[1,1]
    @test_approx_eq util_11 manual_11

    manual_12 = sigma^2 * (1 - rho^2) * (cos(phi)sin(phi))
    util_12 = Util.get_bvn_cov(rho, phi, sigma)[1,2]
    @test_approx_eq util_12 manual_12

    manual_22 = sigma^2 * (1 + (rho^2 - 1) * (cos(phi))^2)
    util_22 = Util.get_bvn_cov(rho, phi, sigma)[2,2]
    @test_approx_eq util_22 manual_22
end


####################################################


#=
test_util_bvn_cov()
test_tiny_image_tiling()
test_kl_divergence_derivs()
test_brightness_derivs()
test_sky_noise_estimates()
test_kl_divergence_values()
test_local_sources_2()
test_local_sources()
test_coordinates_vp_conversion()
test_accum_pos()
test_accum_pixel_source_derivs()
test_elbo_derivs()
test_that_variance_is_low()
test_that_star_truth_is_more_likely()
test_star_optimization()
test_full_elbo_optimization()
test_galaxy_optimization()
=#
test_that_galaxy_truth_is_more_likely()

test_coadd_cat_init_is_more_likely()

#test_real_stamp_optimization()  # long running

# test_tiling()  # bug

#test_peak_init_galaxy_optimization()
#test_peak_init_2body_optimization()
