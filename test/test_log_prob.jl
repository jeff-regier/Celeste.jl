using Celeste: Model
using Base.Test

"""
test log_prob.jl and log_prob_util.jl
"""

#####################
## Helper functions #
#####################


function test_that_star_truth_is_most_likely_log_prob()

    # init ground truth star
    blob, ea, body = true_star_init();
    blob_tiles     = [Model.TiledImage(b; tile_width=b.W) for b in blob]
    active_pixels  = ea.active_pixels

    # turn list of catalog entries a list of LatentStateParams
    source_states = [Model.catalog_entry_to_latent_state_params(body[s])
                     for s in 1:length(body)]

    # create logpdf function handle
    star_logpdf, star_logprior =
        Model.make_star_logpdf(ea.images, active_pixels, ea.S, ea.N,
                               source_states, ea.tile_source_map,
                               ea.patches, ea.active_sources,
                               ea.psf_K, ea.num_allowed_sd)

    # extract the star-specific parameters from source_states[1] for the
    # logpdf function
    star_state = Model.extract_star_state(source_states[1])
    println(star_state)
    best_ll = star_logpdf(star_state)
    println("Best ll ", best_ll)

    # perterb lnr
    lnr = star_state[1]
    for bad_lnr in [lnr*.1, lnr*.5, 1.5*lnr, 2.0*lnr]
        bad_state = deepcopy(star_state)
        bad_state[1] = bad_lnr
        @test best_ll > star_logpdf(bad_state)
    end

    # perturb ra, dec
    ra, dec = star_state[end-1], star_state[end]
    for bad_ra in [.5*ra, .8*ra, 1.1*ra, 1.5*ra]
        bad_state = deepcopy(star_state)
        bad_state[end-1] = bad_ra
        @test best_ll > star_logpdf(bad_state)
    end

    # perturb colors
    for scale in [.5, 2.2]

      # generate a random direction in param space
      r = randn(length(star_state))
      r = r / sqrt(sum(r.*r))

      bad_state = star_state + scale * r
      @test best_ll > star_logpdf(bad_state)
    end
end


function test_that_gal_truth_is_most_likely()
    # TODO run same tests as above, but with galaxy logpdf
    error("unimplemented")
end


function test_color_flux_transform()
    # fluxes --- positive poisson rates
    fluxes = [1., 200., 300., 20., 5.]
    lnr, colors = Model.fluxes_to_colors(fluxes)

    # transform back to colors
    fluxes_back = Model.colors_to_fluxes(lnr, colors)
    for i in 1:length(fluxes)
        @test isapprox(fluxes[i], fluxes_back[i])
    end
end


function test_sigmoid_logit()
    as = [-10., -1., -.001, .001, 1., 10.]
    for a in as

        # test that sigmoid is pushing between 0 and 1
        sig_a = Model.sigmoid(a)
        @test (sig_a >= 0) && (sig_a <= 1)

        # test that logit is equal to inverse sigmoid
        a_prime = Model.logit(sig_a)
        @test isapprox(a, a_prime)

    end
end


function test_gal_shape_constrain()
    con_gal_shape = [.1, .1, .1, .1]
    unc_gal_shape = Model.unconstrain_gal_shape(con_gal_shape)
    back_gal_shape = Model.constrain_gal_shape(unc_gal_shape)

    # test equality of inverse map
    @test length(con_gal_shape) == length(back_gal_shape)
    for i in 1:length(unc_gal_shape)
        @test isapprox(con_gal_shape[i], back_gal_shape[i])
    end
end


####################################
test_sigmoid_logit()
test_gal_shape_constrain()
test_color_flux_transform()
test_that_star_truth_is_most_likely_log_prob()
#test_that_gal_truth_is_most_likely()
