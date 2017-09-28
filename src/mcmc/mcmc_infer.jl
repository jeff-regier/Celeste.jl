#### Celeste Specific MCMC Functions

#"""
#Main MCMC function for a field/catalog.
#Mimics a composition of the AccuracyBenchmarks.run_celeste
#and the AccuracyBenchmark.celeste_to_df functions.
#
#Should return catalog that can be used with `accuracy/score_prediction` and
#`accuracy/score_uncertainty`.
#
#Input:
#  - config : defines a pixel restriction
#  - catalog_entries : 
#  - target_sources : (1, ..., n_sources)
#  - images : ...
#
#Output:
#  - results_df : dataframe with rows ...
#"""
#function run_celeste_mcmc(config::Config,
#                          catalog_entries,
#                          target_sources,
#                          images)
#    # for each source, establish list of neighboring sources
#    neighbor_map = ParallelRun.find_neighbors(target_sources, catalog_entries, images)
#
#    # infer on each source
#    results = one_node_single_infer_mcmc(
#      config,
#      catalog_entries,
#      target_sources,
#      neighbor_map,
#      images,
#    )
#
#    # collect into a single dataframe
#    return results
#end
#
#    star_row, gal_row = summarize_samples(mcmc_results, entry)
#    mcmc_results["star_row"] = star_row
#    mcmc_results["gal_row"]  = gal_row
# 


"""
Runs Annealed Importance Sampling on both the Star and Galaxy posteriors
to estimate the marginal likelihood of each model and posterior samples
simultaneously
"""
function run_ais(entry::CatalogEntry,
                 imgs::Array{Image},
                 patches::Array{SkyPatch, 2},
                 background_images::Array{Array{Float64, 2}, 1},
                 pos_delta::Array{Float64, 1}=[2., 2.];
                 num_samples::Int=10,
                 num_temperatures::Int=50,
                 print_skip::Int=20)
    println("\nRunning AIS on with patch size ", imgs[1].H, "x", imgs[1].W)
    println("  catalog type: ", entry.is_star ? "star" : "galaxy")
    println("   num active pixels ")
    for p in patches
        println("  ... ", sum(p.active_pixel_bitmap))
    end
    #imgs, pos_delta, num_samples, print_skip, num_temperatures = patch_images, [1., 1.], 3, 1, 10

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
    println("star loglike at CATALOG vs PRIOR : ", star_loglike(th_cat), ", ", star_loglike(th_rand))
    println("star logprior at CATALOG vs PRIOR : ", star_logprior(th_cat), ", ", star_logprior(th_rand))
    star_schedule = MCMC.sigmoid_schedule(num_temperatures; rad=4)
    res_star = MCMC.ais_slicesample(star_logpost, star_logprior,
                                    sample_star_prior;
                                    schedule=star_schedule,
                                    num_samps=num_samples,
                                    num_samples_per_step=5)
    lnZ = res_star[:lnZ]
    lnZs = res_star[:lnZ_bootstrap]
    lo, hi = percentile(lnZs, 2.5), percentile(lnZs, 97.5)
    @printf "STAR AIS estimate : %2.4f [%2.3f, %2.3f]\n" lnZ lo hi
    @printf "  CI width : %2.5f \n" (hi-lo)

    ####################
    # run galaxy AIS   #
    ####################
    gal_loglike, gal_logprior, gal_logpost, sample_gal_prior, ra_lim, dec_lim, uniform_to_deg, deg_to_uniform = 
          MCMC.make_gal_inference_functions(imgs, entry;
                                        patches=patches[1, :],
                                        background_images=background_images,
                                        pos_delta=pos_delta)

    gal_schedule = MCMC.sigmoid_schedule(num_temperatures; rad=4)
    res_gal = MCMC.ais_slicesample(gal_logpost, gal_logprior,
                                   sample_gal_prior;
                                   schedule=gal_schedule,
                                   num_samps=num_samples,
                                   num_samples_per_step=5)
    lnZ = res_gal[:lnZ]
    lnZs = res_gal[:lnZ_bootstrap]
    lo, hi = percentile(lnZs, 2.5), percentile(lnZs, 97.5)
    @printf "GAL AIS estimate : %2.4f [%2.3f, %2.3f]\n" lnZ lo hi
    @printf "  CI width : %2.5f \n" (hi-lo)

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
    println("  source p-star = ", exp(ave_pstar))

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
    #star_chain[:objid] = [entry.objid for n in 1:size(star_chain)[1]]
    #gal_chain[:objid]  = [entry.objid for n in 1:size(gal_chain)[1]]
    mcmc_results = Dict("star_samples" => star_chain,
                        "star_lls"     => res_star[:lnZsamps],
                        "star_boostrap"=> res_star[:lnZ_bootstrap],
                        "gal_samples"  => gal_chain,
                        "gal_lls"      => res_gal[:lnZsamps],
                        "gal_bootsrap" => res_gal[:lnZ_bootstrap],
                        "type_samples" => type_chain,
                        "ave_pstar"    => ave_pstar)
    return mcmc_results
end


"""
Run single source MCMC chain
"""
function run_mcmc(entry::CatalogEntry,
                  imgs::Array{Image},
                  patches::Array{SkyPatch, 2},
                  background_images::Array{Array{Float64, 2}, 1},
                  pos_delta::Array{Float64, 1}=[1., 1.];
                  num_samples::Int=500,
                  print_skip::Int=20)
    println("\nRunning mcmc on entry with patch size ",
            imgs[1].H, "x", imgs[1].W)
    println("  catalog type: ", entry.is_star ? "star" : "galaxy")

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
    println("loglike at true initial position: ", star_loglike(th_cat))
    println("loglike at random prior position: ", star_loglike(th_rand))
    println("logprior at true position:        ", star_logprior(th_cat))

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
    println("gal ll at Catalog vs Shifted : ",
        gal_loglike(th_cat), " vs. ", gal_loglike(th_rand))
    println("gal lnprior at true initial pos:  ", gal_logprior(th_cat))
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
    println("  source p-star = ", exp(ave_pstar))

    ####################################################################
    # convert positions to RA/Dec and organize chains into dataframes  #
    ####################################################################
    #for n in 1:size(star_chain)[1]
    #    star_chain[n,6:7] = constrain_pos(star_chain[n,6:7])
    #end
    #for n in 1:size(gal_chain)[1]
    #    gal_chain[n,6:7] = constrain_pos(gal_chain[n,6:7])
    #end
    star_chain = MCMC.samples_to_dataframe(star_chain; is_star=true)
    gal_chain = MCMC.samples_to_dataframe(gal_chain; is_star=false)

    # store objid (for concatenation)
    #star_chain[:objid] = [entry.objid for n in 1:size(star_chain)[1]]
    #gal_chain[:objid]  = [entry.objid for n in 1:size(gal_chain)[1]]
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
function summarize_samples(results_dict, entry)
    stardf   = results_dict["star_samples"]
    galdf    = results_dict["gal_samples"]
    star_row = MCMC.samples_to_dataframe_row(stardf; is_star=true)
    gal_row  = MCMC.samples_to_dataframe_row(galdf; is_star=false)
    return star_row, gal_row
end


"""
Consolidate samples into results dataframes (single entry per source)
"""
function consolidate_samples(reslist)
    stardf    = vcat([r["star_samples"] for r in reslist]...)
    galdf     = vcat([r["gal_samples"] for r in reslist]...)
    star_summary = vcat([r["star_row"] for r in reslist]...)
    gal_summary  = vcat([r["gal_row"] for r in reslist]...)

    # compute average type, and create a unified typed list
    joint_summary = []
    for r in reslist
        pstar = exp(r["ave_pstar"])
        srow  = (pstar > .5) ? r["star_row"] : r["gal_row"]
        srow[:, :is_star] = [pstar]
        push!(joint_summary, srow)
    end
    joint_summary = vcat(joint_summary...)

    return stardf, galdf, star_summary, gal_summary, joint_summary
end


