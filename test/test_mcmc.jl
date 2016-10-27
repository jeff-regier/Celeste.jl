using Base.Test

using Celeste: Model, Transform, SensitiveFloats, MCMC
using StatsBase

#### to run
include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))
import SampleData: gen_sample_star_dataset


function verify_sample_star(vs, pos)
    @test vs[ids.a[2, 1]] <= 0.01

    @test_approx_eq_eps vs[ids.u[1]] pos[1] 0.1
    @test_approx_eq_eps vs[ids.u[2]] pos[2] 0.1

    brightness_hat = exp(vs[ids.r1[1]] + 0.5 * vs[ids.r2[1]])
    @test_approx_eq_eps brightness_hat / sample_star_fluxes[3] 1. 0.01

    true_colors = log(sample_star_fluxes[2:5] ./ sample_star_fluxes[1:4])
    for b in 1:4
        @test_approx_eq_eps vs[ids.c1[b, 1]] true_colors[b] 0.2
    end
end

function verify_sample_galaxy(vs, pos)
    @test vs[ids.a[2, 1]] >= 0.99

    @test_approx_eq_eps vs[ids.u[1]] pos[1] 0.1
    @test_approx_eq_eps vs[ids.u[2]] pos[2] 0.1

    @test_approx_eq_eps vs[ids.e_axis] .7 0.05
    @test_approx_eq_eps vs[ids.e_dev] 0.1 0.08
    @test_approx_eq_eps vs[ids.e_scale] 4. 0.2

    phi_hat = vs[ids.e_angle]
    phi_hat -= floor(phi_hat / pi) * pi
    five_deg = 5 * pi/180
    @test_approx_eq_eps phi_hat pi/4 five_deg

    brightness_hat = exp(vs[ids.r1[2]] + 0.5 * vs[ids.r2[2]])
    @test_approx_eq_eps brightness_hat / sample_galaxy_fluxes[3] 1. 0.01

    true_colors = log(sample_galaxy_fluxes[2:5] ./ sample_galaxy_fluxes[1:4])
    for b in 1:4
        @test_approx_eq_eps vs[ids.c1[b, 2]] true_colors[b] 0.2
    end
end


#########################################################

function test_star_mcmc()
    blob, ea, body = SampleData.gen_sample_star_dataset();

    # run single source slice sampler on synthetic dataset
    star_chain = MCMC.run_single_star_mcmc(200,  # num_samples,
                                           body, # sources
                                           ea.images,
                                           ea.active_pixels, ea.S, ea.N,
                                           ea.tile_source_map,
                                           ea.patches, ea.active_sources,
                                           ea.psf_K, ea.num_allowed_sd)

    # chain stuff
    Mamba.describe(star_chain)

    # make sure chains contain true params in middle 95%
    source_states = [Model.catalog_entry_to_latent_state_params(s)
                     for s in body]
    println("ground truth params", source_states[1])
    true_star_state = Model.extract_star_state(source_states[1])

    # check to make sure posterior percentiles cover truth
    star_param_names = ["lnr", "cug", "cgr", "cri", "ciz", "ra", "dec"]
    for i in 1:length(true_star_state)
        ths = star_chain.value[:,i,1]
        lo, hi = percentile(ths, 1), percentile(ths, 99)
        sin = @printf "   %s   = %2.4f  [%2.4f,  %2.4f] " star_param_names[i] mean(ths) lo hi
        println(sin)
        @test (true_star_state[i] < hi) & (true_star_state[i] > lo)
    end
end


#function test_single_source_optimization()
#    blob, ea, three_bodies = gen_three_body_dataset();
#
#    # Change the tile size.
#    s = 2
#    ea = make_elbo_args(blob, three_bodies, tile_width=10, fit_psf=false, active_source=s);
#    ea_original = deepcopy(ea);
#
#    omitted_ids = Int[]
#    DeterministicVI.elbo_likelihood(ea).v[1]
#    DeterministicVI.maximize_f(DeterministicVI.elbo_likelihood, ea; loc_width=1.0)
#
#    # Test that it only optimized source s
#    @test ea.vp[s] != ea_original.vp[s]
#    for other_s in setdiff(1:ea.S, s)
#        @test_approx_eq ea.vp[other_s] ea_original.vp[other_s]
#    end
#end
#
#function test_galaxy_optimization()
#    # NLOpt fails here so use newton.
#    blob, ea, body = gen_sample_galaxy_dataset();
#    DeterministicVI.maximize_f(DeterministicVI.elbo_likelihood, ea; loc_width=3.0)
#    verify_sample_galaxy(ea.vp[1], [8.5, 9.6])
#end
#
#
#function test_full_elbo_optimization()
#    blob, ea, body = gen_sample_galaxy_dataset(perturb=true);
#    DeterministicVI.maximize_f(DeterministicVI.elbo, ea; loc_width=1.0, xtol_rel=0.0);
#    verify_sample_galaxy(ea.vp[1], [8.5, 9.6]);
#end


test_star_mcmc()
#test_galaxy_mcmc()
#test_single_source_mcmc()
#test_real_stamp_optimization()
