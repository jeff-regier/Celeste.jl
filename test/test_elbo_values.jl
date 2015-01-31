function true_star_init()
    blob, mp, body = gen_sample_star_dataset(perturb=false)

    mp.vp[1][ids.zeta] = 1e-4
    mp.vp[1][ids.gamma] = sample_star_fluxes[3] ./ mp.vp[1][ids.zeta]
    mp.vp[1][ids.lambda] = 1e-4

    blob, mp, body
end

#################################

function test_kl_divergence_values()
	blob, mp, three_bodies = gen_three_body_dataset()

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
	function sklr(accum)
		ElboDeriv.subtract_kl_r!(i, s, mp, accum)
        @assert i == 1
		accum.v /= vs[ids.chi]
	end
    test_kl(q_r, p_r, sklr)

	# k
	q_k = Categorical(vs[ids.kappa[:, i]])
	p_k = Categorical(mp.pp.Psi[i])
	function sklk(accum)
		ElboDeriv.subtract_kl_k!(i, s, mp, accum)
        @assert i == 1
		accum.v /= vs[ids.chi]
	end
	test_kl(q_k, p_k, sklk)

	# c
	mp.pp.Omega[i][:, d] = vs[ids.beta[:, i]]
	mp.pp.Lambda[i][d] = diagm(vs[ids.lambda[:, i]])
	q_c = MvNormal(vs[ids.beta[:, i]], diagm(vs[ids.lambda[:, i]]))
	p_c = MvNormal(mp.pp.Omega[i][:, d], mp.pp.Lambda[i][d])
	function sklc(accum)
		ElboDeriv.subtract_kl_c!(d, i, s, mp, accum)
		accum.v /= vs[ids.chi] * vs[ids.kappa[d, i]]
	end
	test_kl(q_c, p_c, sklc)
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


function test_that_star_truth_is_most_likely()
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


function test_that_galaxy_truth_is_most_likely()
	blob, mp, body = gen_sample_galaxy_dataset(perturb=false)
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


function test_coadd_cat_init_is_most_likely()  # on a real stamp
	stamp_id = "5.0562-0.0643"
	blob = SDSS.load_stamp_blob(dat_dir, stamp_id)
	cat_entries = SDSS.load_stamp_catalog(dat_dir, "s82-$stamp_id", blob)
    bright(ce) = sum(ce.star_fluxes) > 3 || sum(ce.gal_fluxes) > 3
    cat_entries = filter(bright, cat_entries)
    inbounds(ce) = ce.pos[1] > -10. && ce.pos[2] > -10 && 
        ce.pos[1] < 61 && ce.pos[2] < 61
    cat_entries = filter(inbounds, cat_entries)

	mp = ModelInit.cat_init(cat_entries)
	best = ElboDeriv.elbo_likelihood(blob, mp)

	# s is the brightest source: a dev galaxy!
	s = 1
	println(cat_entries[s])

	for bad_scale in [.7, 1.3]
		mp_gamma = deepcopy(mp)
		mp_gamma.vp[s][ids.gamma] *= bad_scale^2
		mp_gamma.vp[s][ids.zeta] /= bad_scale  # keep variance the same
		bad_gamma = ElboDeriv.elbo_likelihood(blob, mp_gamma)
		@test best.v > bad_gamma.v
	end

    for n in [:rho, :phi, :sigma]
        for bad_scale in [.6, 1.8]
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
		for delta in [-2., 2.]
			mp_beta = deepcopy(mp)
			mp_beta.vp[s][ids.beta[b, :]] += delta
			bad_beta = ElboDeriv.elbo_likelihood(blob, mp_beta)
			@test best.v > bad_beta.v
		end
	end
end


function test_tiny_image_tiling()
	blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
	pc = PsfComponent(1./3, zeros(2), 1e-4 * eye(2))
	trivial_psf = [pc, pc, pc]
	pixels = ones(100, 1) * 12
	pixels[98:100, 1] = [1e3, 1e4, 1e5]
	img = Image(3, 1, pixels, 3, blob0[3].wcs, 3., 4, trivial_psf, 1, 1, 1)
	catalog = [sample_ce([100., 1], true),]
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

####################################################

test_tiny_image_tiling()
test_that_variance_is_low()
test_that_star_truth_is_most_likely()
test_that_galaxy_truth_is_most_likely()
test_coadd_cat_init_is_most_likely()
