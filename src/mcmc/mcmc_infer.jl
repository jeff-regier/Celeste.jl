# High level MCMC and AIS inference functions.  These functions
# construct the star and galaxy log likelihood and prior functions, and
# run either Slice-Sampling or Slice-Sampling-within-AIS

"""
Runs Annealed Importance Sampling on both the Star and Galaxy posteriors
to estimate the marginal likelihood of each model and posterior samples
simultaneously
"""
function run_ais(entry::CatalogEntry,
                 imgs::Vector,
                 patches::Array{ImagePatch, 2},
                 background_images::Array{Array{Float64, 2}, 1},
                 pos_delta::Array{Float64, 1}=[2., 2.];
                 num_samples::Int=2,
                 num_temperatures::Int=50,
                 print_skip::Int=20,
                 num_samples_per_chain::Int=25)
    Log.info("\nRunning AIS on with patch size $(imgs[1].H) x $(imgs[1].W)")
    Log.info("  catalog type: $(entry.is_star ? "star" : "galaxy")")
    Log.info("  num images  : $(length(imgs))")
    n_active = sum([sum(p.active_pixel_bitmap) for p in patches[1, :]]) / length(patches[1,:])
    Log.info("  num active pixels per patch: $(n_active)")
    #imgs, pos_delta, num_samples, print_skip, num_temperatures, num_samples_per_chain = patch_images, [1., 1.], 3, 1, 10, 25

    ###################
    # run star MCMC   #
    ###################
    star_loglike, star_logprior, star_logpost, sample_star_prior, ra_lim, dec_lim, uniform_to_deg, deg_to_uniform =
          MCMC.make_star_inference_functions(imgs, entry;
                                        patches=patches[1, :],
                                        background_images=background_images,
                                        pos_delta=pos_delta)

    th_cat = [log.(entry.star_fluxes)..., deg_to_uniform(entry.pos)...]
    th_rand = sample_star_prior()
    Log.info("star loglike at CATALOG vs PRIOR : ", star_loglike(th_cat), ", ", star_loglike(th_rand))
    Log.info("star logprior at CATALOG vs PRIOR : ", star_logprior(th_cat), ", ", star_logprior(th_rand))
    star_schedule = MCMC.sigmoid_schedule(num_temperatures; rad=4)
    res_star = MCMC.ais_slicesample(star_logpost, star_logprior,
                                    sample_star_prior;
                                    schedule=star_schedule,
                                    num_samps=num_samples,
                                    num_samples_per_step=1)
    star_chains, star_chain_lls = [], []
    for i in 1:size(res_star[:zsamps], 2)
        Log.info("    star chain ", i, " of ", size(res_star[:zsamps], 2))
        star_chain, star_lls = MCMC.slicesample_chain(star_logpost,
            res_star[:zsamps][:,1], num_samples_per_chain;
            print_skip=Int64(num_samples_per_chain/5), verbose=false)
        push!(star_chains, star_chain)
        push!(star_chain_lls, star_lls)
    end
    res_star[:zsamps]    = transpose(vcat(star_chains...))
    res_star[:zsamp_lls] = vcat(star_chain_lls...)
    lnZ  = res_star[:lnZ]
    lnZs = res_star[:lnZ_bootstrap]
    lo, hi = percentile(lnZs, 2.5), percentile(lnZs, 97.5)
    Log.info(@sprintf "STAR AIS estimate : %6.3f [%6.3f, %6.3f]\n" lnZ lo hi)
    Log.info(@sprintf "  CI width : %6.5f \n" (hi-lo))

    ####################
    # run galaxy AIS   #
    ####################
    gal_loglike, gal_logprior, gal_logpost, sample_gal_prior, ra_lim, dec_lim, uniform_to_deg, deg_to_uniform =
          MCMC.make_gal_inference_functions(imgs, entry;
                                        patches=patches[1, :],
                                        background_images=background_images,
                                        pos_delta=pos_delta)
    th_cat = MCMC.parameters_from_catalog(entry; is_star=false)
    th_rand = sample_gal_prior()
    Log.info("gal loglike  at CATALOG vs PRIOR : ", gal_loglike(th_cat), ", ", gal_loglike(th_rand))
    Log.info("gal logprior at CATALOG vs PRIOR : ", gal_logprior(th_cat), ", ", gal_logprior(th_rand))
    gal_schedule = MCMC.sigmoid_schedule(num_temperatures; rad=4)
    res_gal = MCMC.ais_slicesample(gal_logpost, gal_logprior,
                                   sample_gal_prior;
                                   schedule=gal_schedule,
                                   num_samps=num_samples,
                                   num_samples_per_step=1)
    gal_chains, gal_chain_lls = [], []
    for i in 1:size(res_gal[:zsamps], 2)
        Log.info("    gal chain ", i, " of ", size(res_gal[:zsamps], 2))
        gal_chain, gal_lls = MCMC.slicesample_chain(gal_logpost,
            res_gal[:zsamps][:,1], num_samples_per_chain;
            print_skip=Int64(num_samples_per_chain/5), verbose=false)
        push!(gal_chains, gal_chain)
        push!(gal_chain_lls, gal_lls)
    end
    res_gal[:zsamps]    = transpose(vcat(gal_chains...))
    res_gal[:zsamp_lls] = vcat(gal_chain_lls...)
    lnZ = res_gal[:lnZ]
    lnZs = res_gal[:lnZ_bootstrap]
    lo, hi = percentile(lnZs, 2.5), percentile(lnZs, 97.5)
    Log.info(@sprintf "GAL AIS estimate : %6.3f [%6.3f, %6.3f]\n" lnZ lo hi)
    Log.info(@sprintf "  CI width : %6.5f \n" (hi-lo))

    #########################################################
    # Compute prob star vs gal based on marginal likelihood #
    #########################################################
    type_chain  = zeros(length(res_gal[:lnZ_bootstrap]))
    # is_star = [.28, .72] vs [.999, .001] vs [.5, .5]
    lnprob_a    = log(.28)
    lnprob_nota = log(.72)
    for n in 1:length(res_gal[:lnZ_bootstrap])
        # normalizing constant is ln p(data | star)
        # so p(star | data) \propto p(data | star) p(star)
        lnprob_star = res_star[:lnZ_bootstrap][n] + lnprob_a
        lnprob_gal  = res_gal[:lnZ_bootstrap][n] + lnprob_nota
        lnsum = Model.logsumexp([lnprob_star, lnprob_gal]) # normalize
        type_chain[n] = lnprob_star - lnsum
    end
    ave_pstar = Model.logsumexp(type_chain) - log(length(type_chain))
    Log.info("  source p-star = ", exp(ave_pstar))

    ####################################################################
    # convert positions to RA/Dec and organize chains into dataframes  #
    ####################################################################
    for n in 1:size(res_gal[:zsamps], 2)
        res_gal[:zsamps][6:7,n]  = uniform_to_deg(res_gal[:zsamps][6:7,n])
        res_star[:zsamps][6:7,n] = uniform_to_deg(res_star[:zsamps][6:7,n])
    end
    star_chain = MCMC.samples_to_dataframe(transpose(res_star[:zsamps]), is_star=true)
    gal_chain  = MCMC.samples_to_dataframe(transpose(res_gal[:zsamps]), is_star=false)

    # store objid (for concatenation)
    mcmc_results = Dict("star_samples" => star_chain,
                        "star_lls"     => res_star[:zsamp_lls], #[res_star[:lnZsamps],
                        "star_bootstrap"=> res_star[:lnZ_bootstrap],
                        "gal_samples"  => gal_chain,
                        "gal_lls"      => res_gal[:zsamp_lls], #res_gal[:lnZsamps],
                        "gal_bootstrap" => res_gal[:lnZ_bootstrap],
                        "type_samples" => type_chain,
                        "ave_pstar"    => ave_pstar)
    return mcmc_results
end


"""
Run single source MCMC chain
"""
function run_mcmc(entry::CatalogEntry,
                  imgs::Vector,
                  patches::Array{ImagePatch, 2},
                  background_images::Array{Array{Float64, 2}, 1},
                  pos_delta::Array{Float64, 1}=[1., 1.];
                  num_samples::Int=500,
                  print_skip::Int=20)
    Log.info("\nRunning mcmc on entry with patch size ",
            imgs[1].H, "x", imgs[1].W)
    Log.info("  catalog type: ", entry.is_star ? "star" : "galaxy")

    # position log prior --- same for both star and galaxy (constrains to a
    # small window around the existing catalog location
    #imgs, pos_delta, num_samples, print_skip = patch_images, [1., 1.], 200, 20
    pos_logprior, ra_lim, dec_lim = MCMC.make_location_prior(
        imgs[1], entry.pos; pos_pixel_delta=[2., 2.])

    ###################
    # run star MCMC   #
    ###################
    star_loglike, constrain_pos, unconstrain_pos =
        MCMC.make_star_loglike(imgs, entry.pos;
                               patches=patches[1,:],
                               background_images=background_images,
                               pos_delta=pos_delta)

    function star_logprior(th)
        lnfluxes, pos = th[1:5], th[6:end]
        return MCMC.logflux_logprior(lnfluxes; is_star=true) + pos_logprior(pos)
    end

    function star_logpost(th)
        return star_loglike(th) + star_logprior(th)
    end

    # log likelihood at data generating parameters, and random prior
    th_cat = [log.(entry.star_fluxes)..., entry.pos...]
    #th_cat = MCMC.parameters_from_catalog(entry, unconstrain_pos; is_star=true)
    #th_rand = [th_cat[1:5]..., [.001, .001]...]
    th_rand = th_cat + .0001*randn(length(th_cat))
    Log.info("loglike at true initial position: ", star_loglike(th_cat))
    Log.info("loglike at random prior position: ", star_loglike(th_rand))
    Log.info("logprior at true position:        ", star_logprior(th_cat))

    # draw MCMC samples
    star_chain, star_lls = MCMC.slicesample_chain(star_logpost, th_cat,
        num_samples; print_skip=print_skip, verbose=false)
    num_warmup = Int(round(.25 * num_samples))
    star_chain = star_chain[num_warmup:end, :]
    star_lls   = star_lls[num_warmup:end]

    ####################
    # run galaxy MCMC  #
    ####################
    gal_loglike, constrain_pos, unconstrain_pos =
        MCMC.make_gal_loglike(imgs, entry.pos;
                              patches=patches[1,:],
                              background_images=background_images,
                              pos_delta=pos_delta)
    gal_logprior = MCMC.make_gal_logprior()

    # test at catalog initialized position vs shifted --- eye test
    th_cat = MCMC.parameters_from_catalog(entry; is_star=false)
    th_rand = th_cat + .00001*randn(length(th_cat))
    Log.info("gal ll at Catalog vs Shifted : ",
        gal_loglike(th_cat), " vs. ", gal_loglike(th_rand))
    Log.info("gal lnprior at true initial pos:  ", gal_logprior(th_cat))
    th_cat[1:5] = th_rand[1:5]

    function gal_logpost(th)
        # catch illegal values in the prior
        llprior = gal_logprior(th)
        if llprior < -1e100
            return llprior
        end
        return gal_loglike(th; print_params=false) + llprior
    end

    # draw MCMC samples
    gal_chain, gal_lls = MCMC.slicesample_chain(
        gal_logpost, th_cat, num_samples; print_skip=print_skip, verbose=false)
    gal_chain = gal_chain[num_warmup:end, :]
    gal_lls   = gal_lls[num_warmup:end]

    #############################################
    # Compute prob star vs gal based on samples #
    #############################################
    type_chain  = zeros(length(gal_lls))
    lnprob_a    = log(.5)
    lnprob_nota = log(1.-.5)
    for n in 1:length(gal_lls)
        # a == 1 full joint prob ln p(pixels | theta^star, a=star) p(theta^star) p(theta^gal) p(a)
        lnjoint_star = star_lls[n] + gal_logprior(gal_chain[n, :]) + lnprob_a

        # a == 0 (galaxy) full joint
        lnjoint_gal = gal_lls[n] + star_logprior(star_chain[n, :]) + lnprob_nota

        # compute log prob of type = star (1)
        lnsum = Model.logsumexp([lnjoint_star, lnjoint_gal])
        type_chain[n] = lnjoint_star - lnsum
    end
    ave_pstar = Model.logsumexp(type_chain) - log(length(gal_lls))
    Log.info("  source p-star = ", exp(ave_pstar))

    ####################################################################
    # convert positions to RA/Dec and organize chains into dataframes  #
    ####################################################################
    star_chain = MCMC.samples_to_dataframe(star_chain; is_star=true)
    gal_chain = MCMC.samples_to_dataframe(gal_chain; is_star=false)

    # store objid (for concatenation)
    mcmc_results = Dict("star_samples" => star_chain,
                        "star_lls"     => star_lls,
                        "gal_samples"  => gal_chain,
                        "gal_lls"      => gal_lls,
                        "type_samples" => type_chain)
    return mcmc_results
end


"""
Turn MCMC results into a single dataframe row that summarizes the
posterior distribution
"""
function summarize_samples(results_dict)
    stardf   = results_dict["star_samples"]
    galdf    = results_dict["gal_samples"]
    star_row = samples_to_dataframe_row(stardf; is_star=true)
    gal_row  = samples_to_dataframe_row(galdf; is_star=false)
    return star_row, gal_row
end


"""
Consolidate samples into results dataframes (single entry per source)
"""
function consolidate_samples(reslist; objids=nothing)
    # give each source a unique label
    if objids==nothing
        objids = ["samp_$(i)" for i in 1:length(reslist)]
    end

    # loop through each result sample list and summarize
    joint_summary, star_summary, gal_summary = [], [], []
    for i in 1:length(reslist)
        r = reslist[i]

        # compute star and gal summaries
        star_row, gal_row = summarize_samples(r)
        star_row[:objid], gal_row[:objid] = objids[i], objids[i]
        push!(star_summary, star_row)
        push!(gal_summary, gal_row)

        # compute star average
        pstar = exp(r["ave_pstar"])
        srow  = (pstar > .5) ? star_row : gal_row
        srow[:, :is_star] = [pstar]
        push!(joint_summary, srow)
    end
    joint_summary = vcat(joint_summary...)
    star_summary  = vcat(star_summary...)
    gal_summary   = vcat(gal_summary...)
    return star_summary, gal_summary, joint_summary
end
