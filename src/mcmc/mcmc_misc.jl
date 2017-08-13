import Distributions: logpdf, Poisson, Normal, MvNormal
using NPZ

################################
# poisson convenience wrappers #
################################

function elementwise_poisson(lam)
    C, H, W = size(lam)
    samp    = Array(Float64, size(lam))
    for c in 1:C, h in 1:H, w in 1:W
        samp[c, h, w] = float(rand(Distributions.Poisson(lam[c, h, w])))
    end
    return samp
end

# output a single poisson log likelihood
function poisson_lnpdf(data, lam)
    return float(logpdf(Distributions.Poisson(lam), data))
end


#######################################################
# save images + plotting stuff for numpy/matplotlib   #
#######################################################

function save_image_array(img_stack, fname)
    npzwrite(fname, img_stack)
end


function save_blob_images(images, fname)
    imgs = permutedims(cat(3, [i.pixels for i in mod_images]...), (3, 1, 2))
    npzwrite(fname, imgs)
end


function save_plot(domain, range, fname; true_val=nothing)
    domain_name = @sprintf "%s-%s" "fin" fname
    range_name  = @sprintf "%s-%s" "fval" fname
    npzwrite(domain_name, domain)
    npzwrite(range_name, range)

    if true_val!=nothing
      tname = @sprintf "%s-%s" "ftrue" fname
      npzwrite(tname, [true_val])
    end
end


"""
Given a function (lnpdf), that takes in a D dimensional vector (th),
this function fixes D-1 of the dimensions of th, and returns a function
handle that only varies with respect to dimension d.
"""
function curry_dim(lnpdf::Function, th0::Vector{Float64}, dim::Int)
    thd0 = th0[dim]

    function clnpdf(thd::Float64)
        th0[dim] = thd
        return lnpdf(th0)
    end

    return clnpdf
end


"""
Function that mimics the AccuracyBenchmark module function
variational_parameters_to_data_frame_row, for direct comparison of ground
truth parameters to VB fit parameters.
"""
function catalog_to_data_frame_row(catalog_entry; objid="truth")
    ce = catalog_entry
    df = DataFrame()
    fluxes = ce.is_star ? ce.star_fluxes : ce.gal_fluxes
    df[:objid]                          = [objid]
    df[:right_ascension_deg]            = [ce.pos[1]]
    df[:declination_deg]                = [ce.pos[2]]
    df[:is_saturated]                   = [false]
    df[:is_star]                        = [ce.is_star]
    df[:de_vaucouleurs_mixture_weight]  = [ce.gal_frac_dev]
    df[:minor_major_axis_ratio]         = [ce.gal_ab]
    df[:half_light_radius_px]           = [ce.gal_scale]
    df[:angle_deg]                      = [ce.gal_angle]
    df[:reference_band_flux_nmgy]       = [fluxes[3]]
    df[:log_reference_band_flux_stderr] = [NaN]
    df[:color_log_ratio_ug]             = [log(fluxes[2]) - log(fluxes[1])]
    df[:color_log_ratio_gr]             = [log(fluxes[3]) - log(fluxes[2])]
    df[:color_log_ratio_ri]             = [log(fluxes[4]) - log(fluxes[3])]
    df[:color_log_ratio_iz]             = [log(fluxes[5]) - log(fluxes[4])]
    df[:color_log_ratio_ug_stderr]      = [NaN]
    df[:color_log_ratio_gr_stderr]      = [NaN]
    df[:color_log_ratio_ri_stderr]      = [NaN]
    df[:color_log_ratio_iz_stderr]      = [NaN]
    return df
end

function samples_to_data_frame_rows(chain; is_star=true)
    df = DataFrame()

    # reference band flux (+ log flux) and colors
    df[:log_reference_band_flux]  = chain[:,3]
    df[:reference_band_flux_nmgy] = exp.(chain[:,3])
    df[:color_log_ratio_ug] = chain[:,2] .- chain[:,1]
    df[:color_log_ratio_gr] = chain[:,3] .- chain[:,2]
    df[:color_log_ratio_ri] = chain[:,4] .- chain[:,3]
    df[:color_log_ratio_iz] = chain[:,5] .- chain[:,4]

    u_samps = [constrain_pos(chain[p,6:7]) for p in 1:size(chain, 1)]
    u_samps = transpose(hcat(u_samps...))

    df[:right_ascension_deg] = u_samps[:,1]
    df[:declination_deg]     = u_samps[:,2]

    return df
end


function samples_to_data_frame_row(sampdf; objid="mcmc")
    """ only for stars right now """
    df = DataFrame()
    df[:objid]                          = [objid]
    df[:right_ascension_deg]            = [mean(sampdf[:right_ascension_deg])]
    df[:declination_deg]                = [mean(sampdf[:declination_deg])]
    df[:is_saturated]                   = [false]
    df[:is_star]                        = [true]
    df[:de_vaucouleurs_mixture_weight]  = [NaN]
    df[:minor_major_axis_ratio]         = [NaN]
    df[:half_light_radius_px]           = [NaN]
    df[:angle_deg]                      = [NaN]
    df[:reference_band_flux_nmgy]       = [mean(sampdf[:reference_band_flux_nmgy])]
    df[:log_reference_band_flux_stderr] = [std(sampdf[:log_reference_band_flux])]
    df[:color_log_ratio_ug]             = [mean(sampdf[:color_log_ratio_ug])]
    df[:color_log_ratio_gr]             = [mean(sampdf[:color_log_ratio_gr])]
    df[:color_log_ratio_ri]             = [mean(sampdf[:color_log_ratio_ri])]
    df[:color_log_ratio_iz]             = [mean(sampdf[:color_log_ratio_iz])]
    df[:color_log_ratio_ug_stderr]      = [std(sampdf[:color_log_ratio_ug])]
    df[:color_log_ratio_gr_stderr]      = [std(sampdf[:color_log_ratio_gr])]
    df[:color_log_ratio_ri_stderr]      = [std(sampdf[:color_log_ratio_ri])]
    df[:color_log_ratio_iz_stderr]      = [std(sampdf[:color_log_ratio_iz])]
 
    return df
end


# run multiple chains
function run_multi_chain_mcmc(lnpdf::Function, th0s)
  chains, logprobs = [], []
  for c in 1:num_chains
    # initialize a star state (around the existing catalog entry)
    println("---- chain ", c, " ----")
    th0 = init_star_params(star_state)
    samples, lls =
      run_mh_sampler_with_warmup(star_logpdf, th0,
                                 num_samples,
                                 num_warmup;
                                 print_skip=print_skip,
                                 keep_warmup=true,
                                 warmup_prop_scale=warmup_prop_scale,
                                 chain_prop_scale=chain_prop_scale)

    push!(chains, samples[(num_warmup+1):end, :])
    push!(logprobs, lls[(num_warmup+1):end])

    # report PSRF after c = 3 chains
    if c > 2
      psrfs = MCMC.potential_scale_reduction_factor(chains)
      println(" potential scale red factor", psrfs)
    end
  end
end


######################################################################
# unit flux image utils --- writes unit flux images to pixel array   #
# Very similar to Synthetic.jl functions                             #
######################################################################

function write_star_unit_flux(img0::Image,
                              pos::Array{Float64, 1},
                              pixels::Matrix{Float64})
    iota = median(img0.iota_vec)
    for k in 1:length(img0.psf)
        the_mean = SVector{2}(WCS.world_to_pix(img0.wcs, pos)) + img0.psf[k].xiBar
        the_cov = img0.psf[k].tauBar
        intensity = iota * img0.psf[k].alphaBar
        write_gaussian(the_mean, the_cov, intensity, pixels)
    end
end

function write_galaxy_unit_flux(img0::Image,
                                pos::Array{Float64,1},
                                gal_frac_dev::Float64,
                                gal_ab::Float64,
                                gal_angle::Float64,
                                gal_scale::Float64,
                                pixels::Matrix{Float64})
    iota = median(img0.iota_vec)
    e_devs = [gal_frac_dev, 1 - gal_frac_dev]
    #XiXi = DeterministicVI.get_bvn_cov(ce.gal_ab, ce.gal_angle, ce.gal_scale)
    XiXi = Model.get_bvn_cov(gal_ab, gal_angle, gal_scale)

    for i in 1:2
        for gproto in galaxy_prototypes[i]
            for k in 1:length(img0.psf)
                the_mean = SVector{2}(WCS.world_to_pix(img0.wcs, pos)) +
                           img0.psf[k].xiBar
                the_cov = img0.psf[k].tauBar + gproto.nuBar * XiXi
                intensity = iota * img0.psf[k].alphaBar * e_devs[i] *
                    gproto.etaBar
                write_gaussian(the_mean, the_cov, intensity, pixels)
            end
        end
    end
end

function write_gaussian(the_mean, the_cov, intensity, pixels)
    the_precision = inv(the_cov)
    c = sqrt(det(the_precision)) / 2pi

    H, W = size(pixels)
    w_range, h_range = get_patch(the_mean, H, W)

    for w in w_range, h in h_range
        y = @SVector [the_mean[1] - h, the_mean[2] - w] # Maybe not hard code Float64
        ypy = dot(y,  the_precision * y)
        pdf_hw = c * exp(-0.5 * ypy)
        pixel_rate = intensity * pdf_hw
        pixels[h, w] += pixel_rate
    end
    pixels
end

function get_patch(the_mean::SVector{2,Float64}, H::Int, W::Int)
    radius = 50
    hm = round(Int, the_mean[1])
    wm = round(Int, the_mean[2])
    w11 = max(1, wm - radius):min(W, wm + radius)
    h11 = max(1, hm - radius):min(H, hm + radius)
    return (w11, h11)
end


#"""
#Main star + galaxy MCMC functions --- creates logpdf and runs
#mcmc sampler
#"""
#function run_single_star_mcmc(sources::Vector{CatalogEntry},
#                              images::Vector{Image},
#                              patches::Matrix{SkyPatch},
#                              active_sources::Vector{Int},
#                              psf_K::Int64;
#                              num_samples::Int64=1000,
#                              num_chains::Int64=5,
#                              num_warmup::Int64=200,
#                              chain_prop_scale::Float64=.05,
#                              warmup_prop_scale::Float64=.1,
#                              print_skip::Int64=250)
#
#    # turn list of catalog entries a list of LatentStateParams
#    # and create logpdf function handle
#    source_states = [Model.catalog_entry_to_latent_state_params(s)
#                     for s in sources]
#    S = length(source_states)
#    N = length(images)
#    star_logpdf, star_logprior =
#        Model.make_star_logpdf(images, S, N, source_states,
#                               patches, active_sources, psf_K)
#
#    # initialize star params
#    star_state = Model.extract_star_state(source_states[1])
#    best_ll = star_logpdf(star_state)
#
#    # determine proposal scale for each dimension --- estimate the
#    # diagonal of the hessian
#
#    # run multiple chains
#    chains, logprobs = [], []
#    for c in 1:num_chains
#      # initialize a star state (around the existing catalog entry)
#      println("---- chain ", c, " ----")
#      th0 = init_star_params(star_state)
#      samples, lls =
#        run_mh_sampler_with_warmup(star_logpdf, th0,
#                                   num_samples,
#                                   num_warmup;
#                                   print_skip=print_skip,
#                                   keep_warmup=true,
#                                   warmup_prop_scale=warmup_prop_scale,
#                                   chain_prop_scale=chain_prop_scale)
#
#      push!(chains, samples[(num_warmup+1):end, :])
#      push!(logprobs, lls[(num_warmup+1):end])
#
#      # report PSRF after c = 3 chains
#      if c > 2
#        psrfs = MCMC.potential_scale_reduction_factor(chains)
#        println(" potential scale red factor", psrfs)
#      end
#    end
#
#    return chains, logprobs
#end
#
#
#function run_single_galaxy_mcmc(sources::Vector{CatalogEntry},
#                                images::Vector{Image},
#                                patches::Matrix{SkyPatch},
#                                active_sources::Vector{Int},
#                                psf_K::Int64;
#                                num_samples::Int64=1000,
#                                num_chains::Int64=5,
#                                num_warmup::Int64=200,
#                                prop_scale::Float64=.01,
#                                print_skip::Int64=250)
#    source_states = [Model.catalog_entry_to_latent_state_params(s)
#                     for s in sources]
#    gal_logpdf, gal_logprior =
#        Model.make_galaxy_logpdf(images, S, N,
#                                 source_states,
#                                 patches, active_sources,
#                                 psf_K)
#
#    # initialize star params
#    gal_state = Model.extract_galaxy_state(source_states[1])
#    println(gal_state)
#    best_ll = gal_logpdf(gal_state)
#
#    # run star slic esampler for 
#    gal_sim = run_slice_sampler(gal_logpdf, gal_state,
#                                 num_samples, galaxy_param_names)
#    gal_sim
#
#end
#
#
#function init_star_params(star_params::Vector{Float64};
#                          radec_scale::Float64=1e-5)
#    th0 = copy(star_params)
#    for ii in 1:5
#      th0[ii] += .1*randn()
#    end
#    th0[6:7] += radec_scale*randn(2)
#    return th0
#end
#
############
## Helpers #
############
#
#"""
#Rudimentary proposal scale determination --- given a log probability
#function, this numerically determines the diagonal of the Hessian, and
#returns its inverse 1/diag(Hessian).
#
#This is meant to capture the relative scale between the different 
#parameters for the log likelihood function, used to determine
#reasonable proposals
#"""
#function compute_proposal_scale(lnpdf::Function,
#                                th0::Vector{Float64})
#    # univariate derivative function generator
#    function numgrad(f::Function, d::Int64)
#        function df(th::Vector{Float64})
#          de      = 1e-6
#          the     = copy(th)
#          the[d] += de
#          return (lnpdf(the) - lnpdf(th)) / de
#        end
#        return df
#    end
#
#    # for each dimension, numerically compute elementwise gradient twice
#    D = length(th0)
#    Hdiag = zeros(D)
#    for d in 1:D
#        df  = numgrad(lnpdf, d)
#        ddf = numgrad(df, d)
#        Hdiag[d] = ddf(th0)
#    end
#    return 1./ sqrt.(abs.(Hdiag))
#end
#
#
#"""
#Run single metropolis hastings chain
#"""
#
#function run_mh_sampler_with_warmup(lnpdf::Function,
#                                    th0::Vector{Float64},
#                                    N::Int64,
#                                    warmup::Int64;
#                                    print_skip::Int64=100,
#                                    keep_warmup::Bool=false,
#                                    warmup_prop_scale::Float64=.1,
#                                    chain_prop_scale::Float64=.05)
#    println("warming up .... ")
#    prop_scale   = warmup_prop_scale*compute_proposal_scale(lnpdf, th0)
#    wchain, wlls = run_mh_sampler(lnpdf, th0, warmup, prop_scale;
#                                  print_skip=print_skip)
#
#    println("running sampler .... ")
#    th0 = wchain[end,:]
#    prop_scale = chain_prop_scale*compute_proposal_scale(lnpdf, th0)
#    chain, lls = run_mh_sampler(lnpdf, th0, N, prop_scale;
#                                print_skip=print_skip)
#
#    if keep_warmup
#        chain, lls = vcat([wchain, chain]...), vcat([wlls, lls]...)
#    end
#    return chain, lls
#end
#
#function run_mh_sampler(lnpdf::Function,
#                        th0::Vector{Float64},
#                        N::Int64,
#                        prop_scale::Vector{Float64};
#                        print_skip::Int=100)
#
#    # stack of samples, log probs, and accepts
#    D = length(th0)
#    samples = zeros(Float64, (N, D))
#    lnprobs = zeros(Float64, N)
#    naccept = 0
#
#    # run chain for N steps
#    thcurr = th0
#    llcurr = lnpdf(thcurr)
#    @printf "  iter : \t loglike \t acc. rat \t num acc. \n"
#    for i in 1:N
#      if mod(i, print_skip) == 0
#          @printf "   %d   : \t %2.4f \t %2.4f \t %d \n" i llcurr (float(naccept)/float(i)) naccept
#      end
#
#      # propose sample
#      thprop = thcurr .+ (prop_scale .* randn(D))
#      llprop = lnpdf(thprop)
#
#      # acceptance ratio
#      aratio = (llprop - llcurr)
#      if log(rand()) < aratio
#          naccept += 1
#          thcurr = thprop
#          llcurr = llprop
#      end
#
#      # store samples
#      samples[i,:] = thcurr
#      lnprobs[i]   = llcurr
#    end
#
#    return samples, lnprobs
#end


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


"""
Convert `chains` (list of sample arrays), `logprobs` (list of 
log-likelihood traces) and `colnames` (list of parameter names for chains)
into a DataFrame object --- for saving and plotting
"""
function chains_to_dataframe(chains, logprobs, colnames)
    # sample, lls, chain identifier all in arrays
    samples    = vcat(chains...)
    sample_lls = vcat(logprobs...)
    chain_id   = vcat([s*ones(Integer, size(chains[s], 1))
                       for s in 1:length(chains)]...)

    # convert to data frame, rename variables
    sdf = convert(DataFrame, samples)
    rename!(sdf, names(sdf), [Symbol(s) for s in colnames])

    # add lls, chain index
    sdf[:lls]   = sample_lls
    sdf[:chain] = chain_id
    return sdf
end


