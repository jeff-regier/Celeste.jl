module MCMC

using ..Model

# TODO move these to model/log_prob.jl
star_param_names = ["lnr", "cug", "cgr", "cri", "ciz", "ra", "dec"]
galaxy_param_names = [star_param_names; ["gdev", "gaxis", "gangle", "gscale"]]


function run_single_star_mcmc(sources::Vector{CatalogEntry},
                              images::Vector{Image},
                              patches::Matrix{SkyPatch},
                              active_sources::Vector{Int},
                              psf_K::Int64;
                              num_samples::Int64=1000,
                              num_chains::Int64=5,
                              num_warmup::Int64=200,
                              prop_scale::Float64=.01,
                              print_skip::Int64=250)

    # make sure we're taking more samples than burnin/warmup
    @assert num_samples > num_warmup

    # turn list of catalog entries a list of LatentStateParams
    # and create logpdf function handle
    source_states = [Model.catalog_entry_to_latent_state_params(s)
                     for s in sources]
    S = length(source_states)
    N = length(images)
    star_logpdf, star_logprior =
        Model.make_star_logpdf(images, S, N, source_states,
                               patches, active_sources, psf_K)

    # initialize star params
    star_state = Model.extract_star_state(source_states[1])
    best_ll = star_logpdf(star_state)

    # run multiple chains
    chains, logprobs = [], []
    for c in 1:num_chains
      # initialize a star state (around the existing catalog entry)
      th0 = init_star_params(star_state)
      samples, lls =
        MCMC.run_mh_sampler(star_logpdf, th0,
                            num_samples + num_warmup,
                            star_param_names;
                            prop_scale=prop_scale,
                            print_skip=print_skip)
      push!(chains, samples[(num_warmup+1):end, :])
      push!(logprobs, lls[(num_warmup+1):end])

      # report PSRF after c = 3 chains
      if c > 2
        psrfs = MCMC.potential_scale_reduction_factor(chains)
        println(" potential scale red factor", psrfs)
      end
    end

    return chains, logprobs
end


function init_star_params(star_params::Vector{Float64};
                          radec_scale::Float64=1e-5)
    th0 = copy(star_params)
    for ii in 1:5
      th0[ii] += .1*randn()
    end
    th0[6:7] += radec_scale*randn(2)
    return th0
end


function run_single_galaxy_mcmc(num_samples::Int64,
                                sources::Vector{CatalogEntry},
                                images::Vector{Image},
                                S::Int64,
                                N::Int64,
                                patches::Matrix{SkyPatch},
                                active_sources::Vector{Int},
                                psf_K::Int64)

    source_states = [Model.catalog_entry_to_latent_state_params(s)
                     for s in sources]
    gal_logpdf, gal_logprior =
        Model.make_galaxy_logpdf(images, S, N,
                                 source_states,
                                 patches, active_sources,
                                 psf_K)

    # initialize star params
    gal_state = Model.extract_galaxy_state(source_states[1])
    println(gal_state)
    best_ll = gal_logpdf(gal_state)

    # run star slic esampler for 
    gal_sim = run_slice_sampler(gal_logpdf, gal_state,
                                 num_samples, galaxy_param_names)
    gal_sim

end


"""
Metropolis Hastings chain
"""
function run_mh_sampler(lnpdf::Function,
                        th0::Vector{Float64},
                        N::Int,
                        param_names::Vector{String};
                        prop_scale::Float64=.1,
                        print_skip::Int=100)

    # stack of samples, log probs, and accepts
    D = length(th0)
    samples = zeros(Float64, (N, D))
    lnprobs = zeros(Float64, N)
    naccept = 0

    # run chain for N steps
    thcurr = th0
    llcurr = lnpdf(thcurr)
    @printf "  iter : \t loglike \t acc. rat \t num acc. \n"
    for i in 1:N
      if mod(i, print_skip) == 0
          @printf "   %d   : \t %2.4f \t %2.4f \t %d \n" i llcurr (float(naccept)/float(i)) naccept
      end

      # propose sample
      thprop = thcurr + prop_scale * randn(D)
      llprop = lnpdf(thprop)

      # acceptance ratio
      aratio = (llprop - llcurr)
      if log(rand()) < aratio
          naccept += 1
          thcurr = thprop
          llcurr = llprop
      end

      # store samples
      samples[i,:] = thcurr
      lnprobs[i]   = llcurr
    end

    return samples, lnprobs
end


"""
Run a slice sampler for N steps
"""
function run_slice_sampler(lnpdf::Function,
                           th0::Vector{Float64},
                           N::Int,
                           param_names::Vector{String})
    # slice sample as in example:
    # http://mambajl.readthedocs.io/en/latest/examples/line_amwg_slice.html
    sim = Mamba.Chains(N, length(th0), names = param_names)
    th  = Mamba.SliceUnivariate(th0, 1., lnpdf)
    for i in 1:N
      if mod(i, 25) == 0
          println("   sample ", i)
      end
      Mamba.sample!(th)
      sim[i, :, 1] = th
    end
    return sim
end


"""
Potential Scale Reduction Factor --- Followed the formula from the 
following website:
http://blog.stata.com/2016/05/26/gelman-rubin-convergence-diagnostic-using-multiple-chains/
"""
function potential_scale_reduction_factor(chains)
    # each chain has to be size N x D, we have M chains
    N, D  = size(chains[1])
    M     = length(chains)

    # mean and variance of each chain
    means = vcat([mean(s, 1) for s in chains]...)
    vars  = vcat([var(s, 1)  for s in chains]...)

    # grand mean
    gmu   = mean(means, 1)

    # between chain variance:w
    B = float(N)/(float(M)-1)*sum( broadcast(-, means, gmu).^2, 1)

    # average within chain variance
    W = mean(vars, 1)

    # compute PRSF ratio
    Vhat = (float(N)-1.)/float(N) * W + (float(M)+1)/float(N*M) * B
    psrf = Vhat ./ W
    return psrf
end


end
