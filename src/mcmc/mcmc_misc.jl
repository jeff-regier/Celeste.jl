import Distributions: logpdf, Poisson, Normal, MvNormal

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
Given a function (lnpdf), that takes in a NUM_COLOR_COMPONENTS dimensional vector (th),
this function fixes NUM_COLOR_COMPONENTS-1 of the dimensions of th, and returns a function
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
    df[:ra]            = [ce.pos[1]]
    df[:dec]                = [ce.pos[2]]
    df[:is_saturated]                   = [false]
    df[:is_star]                        = [ce.is_star]
    df[:gal_frac_dev]  = [ce.gal_frac_dev]
    df[:gal_axis_ratio]         = [ce.gal_axis_ratio]
    df[:gal_radius_px]           = [ce.gal_radius_px]
    df[:gal_angle_deg]                      = [ce.gal_angle]
    df[:flux_r_nmgy]       = [fluxes[3]]
    df[:log_flux_r_stderr] = [NaN]
    df[:color_ug]             = [log(fluxes[2]) - log(fluxes[1])]
    df[:color_gr]             = [log(fluxes[3]) - log(fluxes[2])]
    df[:color_ri]             = [log(fluxes[4]) - log(fluxes[3])]
    df[:color_iz]             = [log(fluxes[5]) - log(fluxes[4])]
    df[:color_ug_stderr]      = [NaN]
    df[:color_gr_stderr]      = [NaN]
    df[:color_ri_stderr]      = [NaN]
    df[:color_iz_stderr]      = [NaN]
    return df
end

function samples_to_data_frame_rows(chain; is_star=true)
    df = DataFrame()

    # reference band flux (+ log flux) and colors
    df[:log_flux_r]  = chain[:,3]
    df[:flux_r_nmgy] = exp.(chain[:,3])
    df[:color_ug] = chain[:,2] .- chain[:,1]
    df[:color_gr] = chain[:,3] .- chain[:,2]
    df[:color_ri] = chain[:,4] .- chain[:,3]
    df[:color_iz] = chain[:,5] .- chain[:,4]

    u_samps = [constrain_pos(chain[p,6:7]) for p in 1:size(chain, 1)]
    u_samps = transpose(hcat(u_samps...))

    df[:ra] = u_samps[:,1]
    df[:dec]     = u_samps[:,2]

    return df
end


function samples_to_data_frame_row(sampdf; objid="mcmc")
    """ only for stars right now """
    df = DataFrame()
    df[:objid]                          = [objid]
    df[:ra]            = [mean(sampdf[:ra])]
    df[:dec]                = [mean(sampdf[:dec])]
    df[:is_saturated]                   = [false]
    df[:is_star]                        = [true]
    df[:gal_frac_dev]  = [NaN]
    df[:gal_axis_ratio]         = [NaN]
    df[:gal_radius_px]           = [NaN]
    df[:gal_angle_deg]                      = [NaN]
    df[:flux_r_nmgy]       = [mean(sampdf[:flux_r_nmgy])]
    df[:log_flux_r_stderr] = [std(sampdf[:log_flux_r])]
    df[:color_ug]             = [mean(sampdf[:color_ug])]
    df[:color_gr]             = [mean(sampdf[:color_gr])]
    df[:color_ri]             = [mean(sampdf[:color_ri])]
    df[:color_iz]             = [mean(sampdf[:color_iz])]
    df[:color_ug_stderr]      = [std(sampdf[:color_ug])]
    df[:color_gr_stderr]      = [std(sampdf[:color_gr])]
    df[:color_ri_stderr]      = [std(sampdf[:color_ri])]
    df[:color_iz_stderr]      = [std(sampdf[:color_iz])]
 
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
    iota = median(img0.nelec_per_nmgy)
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
                                gal_axis_ratio::Float64,
                                gal_angle::Float64,
                                gal_radius_px::Float64,
                                pixels::Matrix{Float64})
    iota = median(img0.nelec_per_nmgy)
    gal_frac_devs = [gal_frac_dev, 1 - gal_frac_dev]
    #XiXi = DeterministicVI.get_bvn_cov(ce.gal_axis_ratio, ce.gal_angle, ce.gal_radius_px)
    XiXi = Model.get_bvn_cov(gal_axis_ratio, gal_angle, gal_radius_px)

    for i in 1:2
        for gproto in galaxy_prototypes[i]
            for k in 1:length(img0.psf)
                the_mean = SVector{2}(WCS.world_to_pix(img0.wcs, pos)) +
                           img0.psf[k].xiBar
                the_cov = img0.psf[k].tauBar + gproto.nuBar * XiXi
                intensity = iota * img0.psf[k].alphaBar * gal_frac_devs[i] *
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


"""
Potential Scale Reduction Factor --- Followed the formula from the 
following website:
http://blog.stata.com/2016/05/26/gelman-rubin-convergence-diagnostic-using-multiple-chains/
"""
function potential_scale_reduction_factor(chains)
    # each chain has to be size N x NUM_COLOR_COMPONENTS, we have M chains
    N, NUM_COLOR_COMPONENTS  = size(chains[1])
    M     = length(chains)

    # mean and variance of each chain
    means = vcat([mean(s, 1) for s in chains]...)
    vars  = vcat([var(s, 1)  for s in chains]...)

    # grand mean
    gmu   = mean(means, 1)

    # between chain variance:w
    NUM_BANDS = float(N)/(float(M)-1)*sum( broadcast(-, means, gmu).^2, 1)

    # average within chain variance
    W = mean(vars, 1)

    # compute PRSF ratio
    Vhat = (float(N)-1.)/float(N) * W + (float(M)+1)/float(N*M) * NUM_BANDS
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


