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
    #lldist = float(logpdf(Distributions.Poisson(lam), data))
    return data*log(lam) - lam - lgamma(data+1.)
end

# constrain checking
function inrange(val, a, b)
    if (val <= a) || (val >= b)
        return false
    end
    return true
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
    df[:ra]            = [ce.pos[1]]
    df[:dec]                = [ce.pos[2]]
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

function samples_to_dataframe(chain; is_star=true)
    """ Turns MCMC samples (star or galaxy chain) into a dataframe of samples
    with parameters we can compare to Photo and VB inferences

    Conversions performed:
      - lnfluxes  : lnfluxes => reference band lnflux and colors
      - gal_angle : radians [0, pi] => degrees [0, 180]
      - half_light_radius_px : the Celeste parameterization is uncoupled from
         the minor_major_axis_ratio --- in the coadd+primary catalogs the
         :half_light_radius_px is scaled by the sqrt of the
         :minor_major_axis_ratio, e.g. px => px / sqrt(gal_ab)
    """
    df = DataFrame()
    # reference band flux (+ log flux) and colors
    df[:log_flux_r]  = chain[:,3]
    df[:flux_r_nmgy] = exp.(chain[:,3])
    df[:color_ug]    = chain[:,2] .- chain[:,1]
    df[:color_gr]    = chain[:,3] .- chain[:,2]
    df[:color_ri]    = chain[:,4] .- chain[:,3]
    df[:color_iz]    = chain[:,5] .- chain[:,4]
    df[:ra]          = chain[:, 6]
    df[:dec]         = chain[:, 7]
    if !is_star
      df[:gal_frac_dev]   = chain[:, 8]
      df[:gal_axis_ratio] = chain[:, 9]
      df[:gal_angle_deg]  = chain[:, 10] * 360 / (2*pi) # rad => deg
      df[:gal_radius_px]  = chain[:, 11] .* sqrt.(df[:gal_axis_ratio])
    end
    return df
end


function samples_to_dataframe_row(sampdf; is_star=true)
    """ summarize a set of samples into a single dataframe row """
    df = DataFrame()
    df[:ra]                = [mean(sampdf[:ra])]
    df[:dec]               = [mean(sampdf[:dec])]
    df[:is_star]           = [true]
    df[:gal_frac_dev]      = [NaN]
    df[:gal_axis_ratio]    = [NaN]
    df[:gal_radius_px]     = [NaN]
    df[:gal_angle_deg]     = [NaN]
    df[:flux_r_nmgy]       = [mean(sampdf[:flux_r_nmgy])]
    df[:log_flux_r]        = [mean(sampdf[:log_flux_r])]
    df[:log_flux_r_stderr] = [std(sampdf[:log_flux_r])]
    df[:color_ug]          = [mean(sampdf[:color_ug])]
    df[:color_gr]          = [mean(sampdf[:color_gr])]
    df[:color_ri]          = [mean(sampdf[:color_ri])]
    df[:color_iz]          = [mean(sampdf[:color_iz])]
    df[:color_ug_stderr]   = [std(sampdf[:color_ug])]
    df[:color_gr_stderr]   = [std(sampdf[:color_gr])]
    df[:color_ri_stderr]   = [std(sampdf[:color_ri])]
    df[:color_iz_stderr]   = [std(sampdf[:color_iz])]
    if !is_star
        df[:is_star]                        = [false]
        df[:gal_frac_dev]   = [mean(sampdf[:gal_frac_dev])]
        df[:gal_axis_ratio] = [mean(sampdf[:gal_axis_ratio])]
        df[:gal_radius_px]  = [mean(sampdf[:gal_radius_px])]
        df[:gal_angle_deg]  = [mean(sampdf[:gal_angle_deg])]
    end
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

"""
Add a star to a matrix of pixels.  Defaults to unit flux.
"""
function write_star_unit_flux(pos::Array{Float64, 1},
                              psf::Array{Model.PsfComponent,1},
                              wcs::WCS.WCSTransform,
                              iota::Float64,
                              pixels::Matrix{Float64};
                              offset::Array{Float64, 1}=[0., 0.],
                              flux::Float64=1.)
    # write the unit-flux scaled, correctly offset psf
    for k in 1:length(psf)
        # mean in pixels space
        the_mean = SVector{2}(WCS.world_to_pix(wcs, pos) - offset) + psf[k].xiBar
        the_cov  = psf[k].tauBar
        intensity = flux * iota * psf[k].alphaBar
        write_gaussian(the_mean, the_cov, intensity, pixels)
    end
end


"""
Add a galaxy model image to a matrix of pixels.  Defaults to unit flux.
"""
function write_galaxy_unit_flux(pos::Array{Float64, 1},
                                psf::Array{Model.PsfComponent,1},
                                wcs::WCS.WCSTransform,
                                iota::Float64,
                                gal_frac_dev::Float64,
                                gal_axis_ratio::Float64,
                                gal_angle::Float64,
                                gal_scale::Float64,
                                pixels::Matrix{Float64};
                                offset::Array{Float64, 1}=[0., 0.],
                                flux::Float64=1.)
    # write the unit-flux scaled, correctly offset galaxy shape +
    # psf convolution
    e_devs = [gal_frac_dev, 1 - gal_frac_dev]
    XiXi = Model.get_bvn_cov(gal_axis_ratio, gal_angle, gal_scale)
    for i in 1:2
        for gproto in galaxy_prototypes[i]
            for k in 1:length(psf)
                the_mean = SVector{2}(WCS.world_to_pix(wcs, pos) - offset) +
                           psf[k].xiBar
                the_cov = psf[k].tauBar + gproto.nuBar * XiXi
                intensity = flux * iota * psf[k].alphaBar * e_devs[i] * gproto.etaBar
                write_gaussian(the_mean, the_cov, intensity, pixels)
            end
        end
    end
end


function write_galaxy_unit_flux_pixel(px_pos::Array{Float64, 1},
                                      psf::Array{Model.PsfComponent,1},
                                      iota::Float64,
                                      gal_frac_dev::Float64,
                                      gal_axis_ratio::Float64,
                                      gal_angle::Float64,
                                      gal_scale::Float64,
                                      pixels::Matrix{Float64};
                                      flux::Float64=1.)
    # write the unit-flux scaled, correctly offset galaxy shape +
    # psf convolution
    e_devs = [gal_frac_dev, 1 - gal_frac_dev]
    XiXi = Model.get_bvn_cov(gal_axis_ratio, gal_angle, gal_scale)
    for i in 1:2
        for gproto in galaxy_prototypes[i]
            for k in 1:length(psf)
                the_mean = px_pos + psf[k].xiBar
                the_cov = psf[k].tauBar + gproto.nuBar * XiXi
                intensity = flux * iota * psf[k].alphaBar * e_devs[i] * gproto.etaBar
                write_gaussian(the_mean, the_cov, intensity, pixels)
            end
        end
    end
end


"""
Generate a model image on a patch, according to that image/patch psf
"""
function render_patch_nmgy(img::Image, patch::ImagePatch, n_bodies::Vector{CatalogEntry})
    # create sky noise background image in nmgy
    patch_pixels = ones(Float32, size(img.sky)) .* img.sky

    # write star/gal model images onto patch_pixels
    for body in n_bodies
        if body.is_star
            Model.write_star_nmgy!(body.pos, body.star_fluxes[img.b], patch,
                patch_pixels; write_to_patch=true)
        else
            Model.write_galaxy_nmgy!(body.pos, body.gal_fluxes[img.b],
                body.gal_frac_dev, body.gal_axis_ratio, body.gal_angle,
                body.gal_radius_px, img.psf, [patch][:,:], patch_pixels;
                write_to_patch=true)
        end
    end
    return Array{Float64}(patch_pixels)
end


"""
Write a gaussian bump on a pixel array
"""
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


"""
creates a new image from an existing larger (field) and a patch object
"""
function patch_to_image(patch::ImagePatch, img::Image; round_pixels_to_int=true)
    # subselect patch pixels from image
    Hr = patch.bitmap_offset[1]:(patch.bitmap_offset[1] +
                                size(patch.active_pixel_bitmap)[1] - 1)
    Wr = patch.bitmap_offset[2]:(patch.bitmap_offset[2] +
                                size(patch.active_pixel_bitmap)[2] - 1)
    patch_pixels = img.pixels[Hr, Wr]
    if round_pixels_to_int
        patch_pixels = round.(patch_pixels)
    end
    H, W = length(Hr), length(Wr)

    # create sub sky intensity object
    sky_small   = fill(median(img.sky.sky_small), H, W)
    calibration = img.sky.calibration[Hr]
    sky_x       = img.sky.sky_x[Hr]
    sky_y       = img.sky.sky_y[Wr]
    sky         = SDSSBackground(sky_small, sky_x, sky_y, calibration)
    nelec_per_nmgy = img.nelec_per_nmgy[Hr]

    # TODO create an adjusted WCS object
    #wcs = deepcopy(img.wcs)
    #wcs[:crpix] = wcs[:crpix] - patch.bitmap_offset

    # instantiate a smaller patch image
    patch_image = Image(patch_pixels, img.b, img.wcs,
                        patch.psf,
                        sky, nelec_per_nmgy, img.psfmap)
    return patch_image
end
