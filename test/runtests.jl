#!/usr/bin/env julia

using Celeste
using CelesteTypes
using Base.Test

using Distributions
	
import Planck
import Synthetic

const stamp_dir = joinpath(Pkg.dir("Celeste"), "dat")


# verify derivatives of fun_to_test by finite differences
function test_by_finite_differences(fun_to_test::Function, mp::ModelParams)
	f::SensitiveFloat = fun_to_test(mp)

	for s in 1:mp.S
		for p1 in 1:length(f.param_index)
			p0 = f.param_index[p1]

			epsilon = 1e-5 / OptimizeElbo.rescaling[p0]
			vp_alt = deepcopy(mp.vp)
			vp_alt[s][p0] += epsilon
			mp_alt = ModelParams(vp_alt, mp.pp, mp.patches, mp.tile_width)

			f_alt::SensitiveFloat = fun_to_test(mp_alt)
			avg_slope = if epsilon > 1.
				f_alt.v / epsilon - f.v / epsilon  # more stable to divide first
			else
				(f_alt.v - f.v) / epsilon  # more stable to subtract first
			end

			d_lb = min(f.d[p1, s], f_alt.d[p1, s]) - 1e-7
			d_ub = max(f.d[p1, s], f_alt.d[p1, s]) + 1e-7
			if abs(d_lb) > 1. && abs(d_ub) > 1.
				d_lb -= 1e-5 * abs(d_lb)
				d_ub += 1e-5 * abs(d_ub)
			end
			if !(d_lb <= avg_slope <= d_ub)
				println("ERROR [source $s, deriv $p1 ($p0)]: $d_lb <= $avg_slope <= $d_ub")
#			else
#				println("PASSED [source $s, deriv $p1 ($p0)]: $d_lb <= $avg_slope <= $d_ub")
			end
			@test d_lb <= avg_slope <= d_ub
			@test d_ub - d_lb < 1e-6 || d_ub / d_lb < 1.0001
		end
	end
end

#########################

function gen_three_body_model()
	srand(1)
	blob0 = SDSS.load_stamp_blob(stamp_dir, "164.4311-39.0359")
	for b in 1:5
		blob0[b].H, blob0[b].W = 112, 238
	end
	brightness7000K = real(Planck.photons_expected(7000., 10., 1e4))
	three_bodies = [
		CatalogGalaxy([4.5, 3.6], brightness7000K , 0.1, [6, 0., 6.]),
		CatalogStar([60.1, 82.2], brightness7000K),
		CatalogGalaxy([71.3, 100.4], brightness7000K , 0.1, [6, 0., 6.]),
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
	brightness7000K = real(Planck.photons_expected(7000., 10., 1e4))
	one_body = CatalogEntry[
		CatalogGalaxy([8.5, 9.6], brightness7000K , 0.1, [6, 0., 6.]),
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
	brightness7000K = real(Planck.photons_expected(7000., 10., 1e4))
	one_body = CatalogEntry[
		CatalogStar([10.1, 12.2], brightness7000K),
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
		E_F = zero_sensitive_float([1], all_params)
		var_F = zero_sensitive_float([1], all_params)
		sb = ElboDeriv.SourceBrightness(mmp.vp[1])
		m_pos = [9, 10.]
		ElboDeriv.accum_pixel_source_stats!(sb, star_mcs, gal_mcs,
			mmp.vp[1], 1, 1, m_pos, 1, fs0m, fs1m, E_F, var_F)
		E_F
	end
	test_by_finite_differences(wrap_apss_ef, mp0)

	function wrap_apss_varf(mmp)
		star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[3].psf, mmp)
		fs0m = zero_sensitive_float([1], star_pos_params)
		fs1m = zero_sensitive_float([1], galaxy_pos_params)
		E_F = zero_sensitive_float([1], all_params)
		var_F = zero_sensitive_float([1], all_params)
		sb = ElboDeriv.SourceBrightness(mmp.vp[1])
		m_pos = [9, 10.]
		ElboDeriv.accum_pixel_source_stats!(sb, star_mcs, gal_mcs,
			mmp.vp[1], 1, 1, m_pos, 3, fs0m, fs1m, E_F, var_F)
		var_F
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

	function test_kl(q_dist, p_dist, subtract_kl_fun!, tol)
		q_samples = rand(q_dist, sample_size)
		empirical_kl = mean(logpdf(q_dist, q_samples) - logpdf(p_dist, q_samples))
		accum = zero_sensitive_float([s], all_params)
		subtract_kl_fun!(accum)
		exact_kl = -accum.v
		@test_approx_eq_eps empirical_kl exact_kl tol
	end

	vs = mp.vp[s]

	# a
	q_a = Bernoulli(vs[ids.chi])
	p_a = Bernoulli(mp.pp.Delta)
	test_kl(q_a, p_a, (accum) -> ElboDeriv.subtract_kl_a!(s, mp, accum), 1e-4)

	# r
	q_r = Gamma(vs[ids.gamma[i]], vs[ids.zeta[i]])
	p_r = Gamma(mp.pp.Upsilon[i], mp.pp.Phi[i])
	test_kl(q_r, p_r, (accum) -> ElboDeriv.subtract_kl_r!(i, s, mp, accum), 1e-3)

	# k
	q_k = Categorical(vs[ids.kappa[:, i]])
	p_k = Categorical(mp.pp.Psi[i])
	test_kl(q_k, p_k, (accum) -> ElboDeriv.subtract_kl_k!(i, s, mp, accum), 1e-2)

	# c
	q_c = MvNormal(vs[ids.beta[:, i]], vs[ids.lambda[:, i]])
	p_c = MvNormal(mp.pp.Omega[i][:, d], mp.pp.Lambda[i][d])
	function sklc(accum)
		ElboDeriv.subtract_kl_c!(d, i, s, mp, accum)
		accum.v /= vs[ids.kappa[d, i]]
	end
	test_kl(q_c, p_c, sklc, 1e-2)
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


function test_that_variance_is_low()
	blob, mp, body = gen_one_star_dataset()

	# very peaked variational distribution---variance for F(m) should be low
	mp.vp[1][ids.mu] = body[1].mu
	mp.vp[1][ids.chi] = 0.01
	mp.vp[1][ids.zeta] = 1e-4
	mp.vp[1][ids.gamma] = body[1].gamma[3] ./ mp.vp[1][ids.zeta]
	mp.vp[1][ids.lambda] = 1e-4
	mp.vp[1][ids.beta[:, 1]] = log(body[1].gamma[2:5] ./ body[1].gamma[1:4])
	mp.vp[1][ids.beta[:, 2]] = log(body[1].gamma[2:5] ./ body[1].gamma[1:4])

	star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[3].psf, mp)
	fs0m = zero_sensitive_float([1], star_pos_params)
	fs1m = zero_sensitive_float([1], galaxy_pos_params)
	E_F = zero_sensitive_float([1], all_params)
	var_F = zero_sensitive_float([1], all_params)
	sb = ElboDeriv.SourceBrightness(mp.vp[1])
	m_pos = [10, 12.]
	ElboDeriv.accum_pixel_source_stats!(sb, star_mcs, gal_mcs,
		mp.vp[1], 1, 1, m_pos, 3, fs0m, fs1m, E_F, var_F)

	@test 0 < sqrt(var_F.v) < 0.1 * E_F.v
end


function test_that_truth_is_more_likely()
	blob, mp, body = gen_one_star_dataset()

	mp.vp[1][ids.mu] = body[1].mu
	mp.vp[1][ids.chi] = 0.01
	mp.vp[1][ids.zeta] = 1e-4
	mp.vp[1][ids.gamma] = body[1].gamma[3] ./ mp.vp[1][ids.zeta]
	mp.vp[1][ids.lambda] = 1e-4
	mp.vp[1][ids.beta[:, 1]] = log(body[1].gamma[2:5] ./ body[1].gamma[1:4])
	mp.vp[1][ids.beta[:, 2]] = log(body[1].gamma[2:5] ./ body[1].gamma[1:4])
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
	blob, mp, body = gen_one_star_dataset()

	mp.vp[1][ids.mu] = body[1].mu
	mp.vp[1][ids.chi] = 0.01
	mp.vp[1][ids.zeta] = 1e-4
	mp.vp[1][ids.gamma] = body[1].gamma[3] ./ mp.vp[1][ids.zeta]
	mp.vp[1][ids.lambda] = 1e-4
	mp.vp[1][ids.beta[:, 1]] = log(body[1].gamma[2:5] ./ body[1].gamma[1:4])
	mp.vp[1][ids.beta[:, 2]] = log(body[1].gamma[2:5] ./ body[1].gamma[1:4])

	OptimizeElbo.maximize_elbo(blob, mp)

	@test_approx_eq mp.vp[1].chi 0.01
	@test_approx_eq_eps mp.vp[1].mu[1] body[1].mu[1] 0.05
	@test_approx_eq_eps mp.vp[1].mu[2] body[1].mu[2] 0.05
end


function test_peak_init_optimization()
	srand(1)
	blob0 = SDSS.load_stamp_blob(stamp_dir, "164.4311-39.0359")

	brightness7000K = real(Planck.photons_expected(7000., 5., 1e4))

	two_bodies = [
		CatalogStar([11.1, 21.2], brightness7000K),
		CatalogGalaxy([15.3, 31.4], brightness7000K , 0.1, [6, 0., 6.]),
	]

   	blob = Synthetic.gen_blob(blob0, two_bodies)
	mp = ModelInit.peak_init(blob) #one giant tile, giant patches
	@test mp.S == 2

	OptimizeElbo.maximize_elbo(blob, mp)

	@test_approx_eq mp.vp[1].chi 0.0001
	@test_approx_eq mp.vp[2].chi 0.9999
	@test_approx_eq_eps mp.vp[1].mu[1] 11.1 0.05
	@test_approx_eq_eps mp.vp[1].mu[2] 21.2 0.05
	@test_approx_eq_eps mp.vp[2].mu[1] 15.3 0.05
	@test_approx_eq_eps mp.vp[2].mu[2] 31.4 0.05
	@test_approx_eq_eps mp.vp[2].Xi[1] 6. 0.2
	@test_approx_eq_eps mp.vp[2].Xi[2] 0. 0.2
	@test_approx_eq_eps mp.vp[2].Xi[3] 6. 0.2
end


function test_small_image()
	srand(1)
	blob0 = SDSS.load_stamp_blob(stamp_dir, "164.4311-39.0359")
	for b in 1:5
		blob0[b].H, blob0[b].W = 100, 200
	end

	brightness7000K = real(Planck.photons_expected(7000., 10., 1e4))

	three_bodies = [
		CatalogStar([10.1, 12.2], brightness7000K),
		CatalogGalaxy([71.3, 100.4], brightness7000K , 0.1, [6, 0., 6.]),
		CatalogGalaxy([81.5, 103.6], brightness7000K , 0.1, [6, 0., 6.]),
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

	brightness7000K = real(Planck.photons_expected(7000., 10., 1e4))

	three_bodies = [
		CatalogGalaxy([4.5, 3.6], brightness7000K , 0.1, [6, 0., 6.]),
		CatalogStar([60.1, 82.2], brightness7000K),
		CatalogGalaxy([71.3, 100.4], brightness7000K , 0.1, [6, 0., 6.]),
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
	brightness7000K = real(Planck.photons_expected(7000., 10., 1e4))
	one_body = CatalogEntry[CatalogStar([50., 50.], brightness7000K),]

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


test_accum_pixel_source_stats()
test_that_variance_is_low()
test_that_truth_is_more_likely()
test_kl_divergence_derivs()
test_kl_divergence_values()
test_accum_pos()
test_brightness_derivs()
test_local_sources_2()
test_local_sources()
test_elbo_likelihood_derivs()

test_optimization_with_good_initialization()
test_peak_init_optimization()
#=
test_tiling()
test_small_image()
=#
