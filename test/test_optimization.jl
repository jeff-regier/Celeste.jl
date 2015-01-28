
function verify_sample_star(vs, pos)
	@test_approx_eq vs[ids.chi] 0.0001

	@test_approx_eq_eps vs[ids.mu[1]] pos[1] 0.1
	@test_approx_eq_eps vs[ids.mu[2]] pos[2] 0.1

    brightness_hat = vs[ids.gamma[1]] * vs[ids.zeta[1]]
	@test_approx_eq_eps brightness_hat / sample_star_fluxes[3] 1. 0.01

	true_colors = log(sample_star_fluxes[2:5] ./ sample_star_fluxes[1:4])
	for b in 1:4
		@test_approx_eq_eps vs[ids.beta[b, 1]] true_colors[b] 0.1
	end
end

function verify_sample_galaxy(vs, pos)
	@test_approx_eq vs[ids.chi] 0.9999

	@test_approx_eq_eps vs[ids.mu[1]] pos[1] 0.1
	@test_approx_eq_eps vs[ids.mu[2]] pos[2] 0.1

	@test_approx_eq_eps vs[ids.rho] .7 0.05
	@test_approx_eq_eps vs[ids.theta] 0.1 0.08
	@test_approx_eq_eps vs[ids.sigma] 4. 0.2

    phi_hat = vs[ids.phi]
    phi_hat -= floor(phi_hat / pi) * pi
    five_deg = 5 * pi/180
	@test_approx_eq_eps phi_hat pi/4 five_deg

    brightness_hat = vs[ids.gamma[2]] * vs[ids.zeta[2]]
	@test_approx_eq_eps brightness_hat / sample_galaxy_fluxes[3] 1. 0.01

	true_colors = log(sample_galaxy_fluxes[2:5] ./ sample_galaxy_fluxes[1:4])
	for b in 1:4
		@test_approx_eq_eps vs[ids.beta[b, 2]] true_colors[b] 0.1
	end
end


#########################################################


function test_star_optimization()
	blob, mp, body = gen_sample_star_dataset()
	OptimizeElbo.maximize_likelihood(blob, mp)
    verify_sample_star(mp.vp[1], [10.1, 12.2])
end


function test_galaxy_optimization()
	blob, mp, body = gen_sample_galaxy_dataset()
	OptimizeElbo.maximize_likelihood(blob, mp)
    verify_sample_galaxy(mp.vp[1], [8.5, 9.6])
end


function test_peak_init_galaxy_optimization()
	blob, mp, body = gen_sample_galaxy_dataset()
	mp = ModelInit.peak_init(blob)
	OptimizeElbo.maximize_likelihood(blob, mp)
    verify_sample_galaxy(mp.vp[1], [8.5, 9.6])
end


function test_peak_init_2body_optimization()
	srand(1)
	blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")

	two_bodies = [
		sample_ce([11.1, 21.2], true),
		sample_ce([15.3, 31.4], false),
	]

   	blob = Synthetic.gen_blob(blob0, two_bodies)
	mp = ModelInit.peak_init(blob) #one giant tile, giant patches
	@test mp.S == 2

	OptimizeElbo.maximize_likelihood(blob, mp)

    verify_sample_star(mp.vp[1], [11.1, 21.2])
    verify_sample_galaxy(mp.vp[2], [15.3, 31.4])
end


function test_full_elbo_optimization()
	blob, mp, body = gen_sample_galaxy_dataset(perturb=true)
	OptimizeElbo.maximize_elbo(blob, mp)
    verify_sample_galaxy(mp.vp[1], [8.5, 9.6])
end


function test_real_stamp_optimization()
	blob = SDSS.load_stamp_blob(dat_dir, "5.0562-0.0643")
	cat_entries = SDSS.load_stamp_catalog(dat_dir, "s82-5.0562-0.0643", blob)
    bright(ce) = sum(ce.star_fluxes) > 3 || sum(ce.gal_fluxes) > 3
    cat_entries = filter(bright, cat_entries)
    inbounds(ce) = ce.pos[1] > -10. && ce.pos[2] > -10 && 
        ce.pos[1] < 61 && ce.pos[2] < 61
    cat_entries = filter(inbounds, cat_entries)

	for ce in cat_entries
		println(ce)
	end

	mp = ModelInit.cat_init(cat_entries)
	OptimizeElbo.maximize_elbo(blob, mp)
end

####################################################

test_star_optimization()
test_full_elbo_optimization()
test_galaxy_optimization()
test_real_stamp_optimization()  # long running
#test_peak_init_2body_optimization()
#test_peak_init_galaxy_optimization()
