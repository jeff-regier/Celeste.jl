using Base.Test

using Celeste: Model, Transform, SensitiveFloats, MCMC
using StatsBase

include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
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

    ##########################
    # load up star lnpdf     #
    ##########################
    # init ground truth star
    ea, vp, catalog = SampleData.true_star_init()

    # run chains
    chains, logprobs =
        MCMC.run_single_star_mcmc(catalog, ea.images, ea.patches,
                                  ea.active_sources, ea.psf_K;
                                  num_samples = 5000,
                                  num_chains  = 6,
                                  prop_scale  = .0005,
                                  print_skip  = 250)

    # make sure chains contain true params in middle 95%
    source_states = [Model.catalog_entry_to_latent_state_params(s)
                     for s in catalog]
    true_star_state = Model.extract_star_state(source_states[1])

    samples = vcat(chains...)
    for i in 1:length(true_star_state)
        ths = samples[:,i] #star_chain.value[:,i,1]
        lo, hi = percentile(ths, 1), percentile(ths, 99)
        @printf "   %s (true_val %2.4f)  = %2.4f  [%2.4f,  %2.4f] \n" MCMC.star_param_names[i] true_star_state[i] mean(ths) lo hi
        @test (true_star_state[i] < hi) & (true_star_state[i] > lo)
    end

end



#function test_galaxy_mcmc()
#    blob, ea, body = SampleData.gen_sample_galaxy_dataset();
#
#    # run single source slice sampler on synthetic dataset
#    gal_chain = MCMC.run_single_galaxy_mcmc(500,  # num_samples,
#                                            body, # sources
#                                            ea.images,
#                                            ea.active_pixels, ea.S, ea.N,
#                                            ea.tile_source_map,
#                                            ea.patches, ea.active_sources,
#                                            ea.psf_K, ea.num_allowed_sd)
#
#    # chain stuff
#    Mamba.describe(gal_chain)
#
#    # make sure chains contain true params in middle 95%
#    source_states = [Model.catalog_entry_to_latent_state_params(s)
#                     for s in body]
#    true_gal_state = Model.extract_galaxy_state(source_states[1])
#
#    # check to make sure posterior percentiles cover truth
#    gal_param_names = MCMC.galaxy_param_names
#    los = Array(Float64, length(true_gal_state))
#    his = Array(Float64, length(true_gal_state))
#    for i in 1:length(true_gal_state)
#        ths = gal_chain.value[:,i,1]
#        los[i], his[i] = percentile(ths, .01), percentile(ths, 99.9)
#        @printf "   %s (true_val %2.4f) = %2.4f  [%2.4f,  %2.4f] \n" gal_param_names[i] true_gal_state[i] mean(ths) lo hi
#    end
#
#    for i in 1:length(true_gal_state)
#        @test (true_gal_state[i] < his[i]) & (true_gal_state[i] > los[i])
#    end
#
#end


#################################################


test_star_mcmc()
#test_galaxy_mcmc()
