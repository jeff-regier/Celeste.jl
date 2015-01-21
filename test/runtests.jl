#!/usr/bin/env julia

using Celeste
using CelesteTypes
using Base.Test

using Distributions
	
import Synthetic

const stamp_dir = joinpath(Pkg.dir("Celeste"), "dat")


# verify derivatives of fun_to_test by finite differences
function test_by_finite_differences(fun_to_test::Function, mp::ModelParams)
	f::SensitiveFloat = fun_to_test(mp)

	for s in 1:mp.S
		for p1 in 1:length(f.param_index)
			p0 = f.param_index[p1]

			basically_flat = abs(f.d[p1, s]) < 1e-5

			# if not flat, a step size epsilon that changes f by about 1e-6...
			epsilon = basically_flat ? 1e-6 : 1e-2 / f.d[p1, s] 

			vp_greater = deepcopy(mp.vp)
			vp_greater[s][p0] += epsilon
			mp_greater = ModelParams(vp_greater, mp.pp, mp.patches, mp.tile_width)
			f_greater::SensitiveFloat = fun_to_test(mp_greater)

			vp_lesser = deepcopy(mp.vp)
			vp_lesser[s][p0] -= epsilon
			mp_lesser = ModelParams(vp_lesser, mp.pp, mp.patches, mp.tile_width)
			f_lesser = fun_to_test(mp_lesser)

			delta = f_greater.v - f_lesser.v
			# for numerical stability...
			avg_slope = abs(f.v) > 1. && abs(2epsilon) > 1. ?
				f_greater.v / 2epsilon - f_lesser.v / 2epsilon :
				delta / 2epsilon

			if basically_flat
				@test_approx_eq_eps f_lesser.v f_greater.v 2e-11
			else
#				println("avg slope: $avg_slope :  f.d = $(f_lesser.d[p1,s])  f.d = $(f.d[p1,s])     f.d = $(f_greater.d[p1,s])")
#				println(1. - avg_slope / f.d[p1, s])
#				@test 1. - avg_slope / f.d[p1, s] < 1e-1

				dl = min(f_lesser.d[p1, s], f_greater.d[p1, s])
				dl -= 1e-6 + (1e-4)abs(dl)
				du = max(f_lesser.d[p1, s], f_greater.d[p1, s])
				du += 1e-6 + (1e-4)abs(du)
#				println(dl, "  ", avg_slope, "  ", du)
				@test (dl <= avg_slope <= du)
			end
		end
	end
end

#########################

const star_fluxes = [4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]

function gen_three_body_model()
	srand(1)
	blob0 = SDSS.load_stamp_blob(stamp_dir, "164.4311-39.0359")
	for b in 1:5
		blob0[b].H, blob0[b].W = 112, 238
	end
	three_bodies = [
		CatalogGalaxy([4.5, 3.6], star_fluxes , 0.1, [6, 0., 6.]),
		CatalogStar([60.1, 82.2], star_fluxes),
		CatalogGalaxy([71.3, 100.4], star_fluxes , 0.1, [6, 0., 6.]),
	]
   	blob = Synthetic.gen_blob(blob0, three_bodies)
	mp = ModelInit.cat_init(three_bodies)

	blob, mp, three_bodies
end

function gen_one_galaxy_dataset()
	srand(1)
	blob0 = SDSS.load_stamp_blob(stamp_dir, "164.4311-39.0359")
	for b in 1:5
		blob0[b].H, blob0[b].W = 20, 23
	end
	one_body = CatalogEntry[
		CatalogGalaxy([8.5, 9.6], star_fluxes , 0.1, [6, 0., 6.]),
	]
   	blob = Synthetic.gen_blob(blob0, one_body)
	mp = ModelInit.cat_init(one_body)

	blob, mp, one_body
end

function gen_one_star_dataset()
	srand(1)
	blob0 = SDSS.load_stamp_blob(stamp_dir, "164.4311-39.0359")
	for b in 1:5
		blob0[b].H, blob0[b].W = 20, 23
	end
	one_body = CatalogEntry[
		CatalogStar([10.1, 12.2], star_fluxes),
	]
   	blob = Synthetic.gen_blob(blob0, one_body)
	mp = ModelInit.cat_init(one_body)

	blob, mp, one_body
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
		ElboDeriv.accum_galaxy_pos!(gal_mcs[1,1,1,1], [9, 10.],
			mmp.vp[1][ids.theta], 1., galaxy_prototypes[1][1].sigmaTilde, 
			mmp.vp[1][ids.Xi], fs1m)
		fs1m
	end
	test_by_finite_differences(wrap_galaxy, mp)
end


function test_accum_pixel_source_stats()
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

function test_elbo_likelihood_derivs()
	blob, mp0, body = gen_one_galaxy_dataset()

	function wrap_likelihood_b1(mmp)
		ElboDeriv.elbo_likelihood([blob[1]], mmp)
	end
	test_by_finite_differences(wrap_likelihood_b1, mp0)

	function wrap_likelihood_b5(mmp)
		ElboDeriv.elbo_likelihood([blob[5]], mmp)
	end
	test_by_finite_differences(wrap_likelihood_b5, mp0)
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
	q_c = MvNormal(vs[ids.beta[:, i]], vs[ids.lambda[:, i]])
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


function truth_init()
	blob, mp, body = gen_one_star_dataset()

	flx = body[1].fluxes
	colors = log(flx[2:5] ./ flx[1:4])

	mp.vp[1][ids.mu] = body[1].pos
	mp.vp[1][ids.chi] = 1e-3
	mp.vp[1][ids.zeta] = 1e-4
	mp.vp[1][ids.gamma] = 1e6#flx[3] ./ mp.vp[1][ids.zeta]
	mp.vp[1][ids.lambda] = 1e-4
	mp.vp[1][ids.beta[:, 1]] = mp.vp[1][ids.beta[:, 2]] = colors

	blob, mp, body
end


function test_that_variance_is_low()
	# very peaked variational distribution---variance for F(m) should be low
	blob, mp, body = truth_init()

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


function test_that_truth_is_more_likely()
	blob, mp, body = truth_init()
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
			mp_beta.vp[1][ids.beta[b]] *= delta
			bad_beta = ElboDeriv.elbo_likelihood(blob, mp_beta)
			@test best.v > bad_beta.v
		end
	end
end


function test_optimization_with_good_initialization()
	blob, mp, body = truth_init()
	
	OptimizeElbo.maximize_elbo(blob, mp)

	@test_approx_eq mp.vp[1][ids.chi] 0.0001
	@test_approx_eq_eps mp.vp[1][ids.mu[1]] 10.1 0.05
	@test_approx_eq_eps mp.vp[1][ids.mu[2]] 12.2 0.05
	flx = body[1].fluxes
	true_colors = log(flx[2:5] ./ flx[1:4])
	for b in 1:4
		@test_approx_eq_eps mp.vp[1][ids.beta[b, 1]] true_colors[b] 0.1
	end
end


function test_peak_init_optimization()
	srand(1)
	blob0 = SDSS.load_stamp_blob(stamp_dir, "164.4311-39.0359")

	two_bodies = [
		CatalogStar([11.1, 21.2], star_fluxes),
		CatalogGalaxy([15.3, 31.4], star_fluxes , 0.1, [6, 0., 6.]),
	]

   	blob = Synthetic.gen_blob(blob0, two_bodies)
	mp = ModelInit.peak_init(blob) #one giant tile, giant patches
	@test mp.S == 2

	OptimizeElbo.maximize_elbo(blob, mp)

	@test_approx_eq mp.vp[1][ids.chi] 0.0001
	@test_approx_eq mp.vp[2][ids.chi] 0.9999
	@test_approx_eq_eps mp.vp[1][ids.mu[1]] 11.1 0.05
	@test_approx_eq_eps mp.vp[1][ids.mu[2]] 21.2 0.05
	@test_approx_eq_eps mp.vp[2][ids.mu[1]] 15.3 0.05
	@test_approx_eq_eps mp.vp[2][ids.mu[2]] 31.4 0.05
	@test_approx_eq_eps mp.vp[2][ids.Xi[1]] 6. 0.2
	@test_approx_eq_eps mp.vp[2][ids.Xi[2]] 0. 0.2
	@test_approx_eq_eps mp.vp[2][ids.Xi[3]] 6. 0.2
end


function test_small_image()
	srand(1)
	blob0 = SDSS.load_stamp_blob(stamp_dir, "164.4311-39.0359")
	for b in 1:5
		blob0[b].H, blob0[b].W = 100, 200
	end

	three_bodies = [
		CatalogStar([10.1, 12.2], star_fluxes),
		CatalogGalaxy([71.3, 100.4], star_fluxes , 0.1, [6, 0., 6.]),
		CatalogGalaxy([81.5, 103.6], star_fluxes , 0.1, [6, 0., 6.]),
	]

   	blob = Synthetic.gen_blob(blob0, three_bodies)
	mp = ModelInit.peak_init(blob)
	@test mp.S == 3

	elbo = ElboDeriv.elbo(blob, mp)

	@test_approx_eq elbo.v -1.0817836180574356e7
end


function test_local_sources()
	srand(1)
	blob0 = SDSS.load_stamp_blob(stamp_dir, "164.4311-39.0359")
	for b in 1:5
		blob0[b].H, blob0[b].W = 112, 238
	end

	three_bodies = [
		CatalogGalaxy([4.5, 3.6], star_fluxes , 0.1, [6, 0., 6.]),
		CatalogStar([60.1, 82.2], star_fluxes),
		CatalogGalaxy([71.3, 100.4], star_fluxes , 0.1, [6, 0., 6.]),
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
	blob0 = SDSS.load_stamp_blob(stamp_dir, "164.4311-39.0359")
	one_body = CatalogEntry[CatalogStar([50., 50.], star_fluxes),]

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
	blob, mp, three_bodies = gen_three_body_model()
	@test mp.S == 3
	elbo = ElboDeriv.elbo(blob, mp)

	@test_approx_eq_eps elbo.v -9.449959518857952e6 1e-2
	truth = [-508844.6317699191,-433.93385620506535,17282.741853376505,
		0.6544622275104429,1.3266708392392763,2.108588044705838,
		6.003884511466467,0.0675725062900174,0.0,0.684793716373134,
		1.4286385448901462,2.19370988751766,5.812194232750391,
		0.1370825827342583,27674.294319985074,-106680.72303279316,
		322.99952935271295,-22392.660293056626]
	for i in 1:18
		@test_approx_eq_eps elbo.d[i, 1 + i % 3] truth[i] 1e-5
	end

	mp2 = ModelInit.cat_init(three_bodies, tile_width=10)
	elbo_tiles = ElboDeriv.elbo(blob, mp2)
	@test_approx_eq_eps elbo_tiles.v -9.449959518857952e6 1e-2
	@test_approx_eq elbo_tiles.v elbo.v
	for i in 1:18
		@test_approx_eq_eps elbo_tiles.d[i, 1 + i % 3] truth[i] 1e-5
		@test_approx_eq elbo_tiles.d[i, 1 + i % 3] elbo.d[i, 1 + i % 3]
	end

	mp3 = ModelInit.cat_init(three_bodies, patch_radius=30.)
	elbo_patches = ElboDeriv.elbo(blob, mp3)
	@test_approx_eq_eps elbo_patches.v -9.449959518857952e6 1e-2
	for i in 1:18
		@test_approx_eq_eps elbo_patches.d[i, 1 + i % 3] truth[i] 1e-5
	end

	mp4 = ModelInit.cat_init(three_bodies, patch_radius=35., tile_width=10)
	elbo_both = ElboDeriv.elbo(blob, mp4)
	@test_approx_eq_eps elbo_both.v -9.449959518857952e6 1e-1
	for i in 1:18
		tol = abs(truth[i]) * 1e-5
		@test_approx_eq_eps elbo_both.d[i, 1 + i % 3] truth[i] tol
	end
end


function test_sky_noise_estimates()
	blobs = Array(Blob, 2)
	blobs[1], mp, three_bodies = gen_three_body_model()  # synthetic
	blobs[2] = SDSS.load_stamp_blob(stamp_dir, "164.4311-39.0359")  # real

	for blob in blobs
		for b in 1:5
			sdss_sky_estimate = blob[b].epsilon * blob[b].iota
			crude_estimate = median(blob[b].pixels)
			@test_approx_eq_eps sdss_sky_estimate / crude_estimate 1. .3
		end
	end
end


test_that_variance_is_low()
test_sky_noise_estimates()
test_kl_divergence_values()
test_kl_divergence_derivs()
test_accum_pos()
test_local_sources_2()
test_local_sources()
test_accum_pixel_source_stats()
test_elbo_likelihood_derivs()
test_brightness_derivs()
test_that_truth_is_more_likely()
test_optimization_with_good_initialization()
test_peak_init_optimization()

#=
test_tiling()
test_small_image()
=#
