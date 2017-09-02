"""
Star log likelihood maker.  Creates a star log prob function that is
parameterized by a flat vector

  - th : [log_fluxes; unconstrained_pos]

Args:
  imgs: Array of observed data Images with the .pixel field
  pos0: Initial location of the source in ra/dec
  pos_delta: (optional) determines how much ra/dec we allow the sampler
    to drift away from pos0
"""
function make_star_loglike(imgs::Array{Image},
                           pos0::Array{Float64, 1};
                           pos_delta::Array{Float64, 1}=[1., 1.],
                           patches::Array{SkyPatch, 1}=nothing,
                           background_images::Array{Array{Float64, 2}, 1}=nothing)

    # create a function that constrains the pixel location to
    # be within pos0 +- [pos_delta]
    constrain_pos, unconstrain_pos =
        make_position_transformations(pos0, pos_delta)

    # create background images --- sky noise and neighbors if there
    if background_images == nothing
        background_images = make_empty_background_images(imgs)
    end

    # patch offset (0 if no patches passed in)
    if patches == nothing
        offsets = zeros(2, length(imgs))
        active_bitmaps = [BitArray(ones(img.H, img.W)) for img in imgs]
    else
        offsets = hcat([convert(Array{Float64, 1}, p.bitmap_offset)
                        for p in patches]...)
        active_bitmaps = [p.active_pixel_bitmap for p in patches]
    end

    # create function to return
    function star_loglike(th::Array{Float64, 1})
        # unpack log fluxes and location (ra, dec)
        #lnfluxes, unc_pos = th[1:5], th[6:end]
        #pos = constrain_pos(unc_pos)
        lnfluxes, pos = th[1:5], th[6:end]

        ll = 0.
        for ii in 1:length(imgs)
            img, active_bitmap = imgs[ii], active_bitmaps[ii]

            # sky pixel intensity (sky image)
            background = background_images[ii]

            # create and cache unit flux src image
            src_pixels = zeros(img.H, img.W)
            write_star_unit_flux(pos, img.psf, img.wcs,
                                 Float64(median(img.nelec_per_nmgy)), src_pixels,
                                 offset=offsets[:,ii])

            # band-specific flux --- do bounds check
            bflux = exp(lnfluxes[img.b])
            if isinf(bflux)  # if bflux overflows, then return -Inf logprob
                return -Inf
            end

            # sum per-pixel likelihood contribution
            for h in 1:img.H, w in 1:img.W
                pixel_data = img.pixels[h,w]
                is_active  = active_bitmap[h, w]
                if !isnan(pixel_data) && is_active
                    rate_hw = background[h,w] + bflux*src_pixels[h,w]
                    ll += poisson_lnpdf(pixel_data, rate_hw)
                end
            end
        end
        return ll
    end

    return star_loglike, constrain_pos, unconstrain_pos
end


"""
Galaxy log likelihood maker.  Creates a galaxy log prob function that is
parameterized by a flat vector

  - th : [log_fluxes, unconstrained_pos, unconstrained_gal_shape]

Args:
  imgs: Array of observed data Images with the .pixel field
  pos0: Initial location of the source in ra/dec
  pos_delta: (optional) determines how much ra/dec we allow the sampler
    to drift away from pos0

Note on Galaxy Shapes: the `gal_scale` parameter is in pixels.  The 
  Celeste parameterization sigma^2_{cel} is slightly different from the
  Photo and SDSS parameterization for :half_light_radius_px (sigma^2_{sdss}).
  The two have the following relationship:

      sigma^2_{cel}  = sigma^2_{sdss} / sqrt(gal_ab)
   => sigma^2_{sdss} = sigma^2_{cel} * sqrt(gal_ab)

  The function `write_galaxy_unit_flux` is defined for sigma^2_{cel}, that is
  a scale parameter in pixels that indicates the length of the major axis
  (as given by the bivariate_normals.jl#get_bvn_cov function).

  Also note that the scale prior defined below is over sigma^2_{cel}.
"""
function make_gal_loglike(imgs::Array{Image},
                          pos0::Array{Float64, 1};
                          pos_delta::Array{Float64, 1}=[1., 1.],
                          patches::Array{SkyPatch, 1}=nothing,
                          background_images::Array{Array{Float64, 2}, 1}=nothing)
    # create a function that constrains the pixel location to be within
    # pos0 +- [pos_delta]
    constrain_pos, unconstrain_pos =
        make_position_transformations(pos0, pos_delta)

    # create background images --- sky noise and neighbors if there
    if background_images == nothing
        background_images = make_empty_background_images(imgs)
    end

    # patch offset (0 if no patches passed in)
    if patches == nothing
        offsets = zeros(2, length(imgs))
        active_bitmaps = [BitArray(ones(img.H, img.W)) for img in imgs]
    else
        offsets = hcat([convert(Array{Float64, 1}, p.bitmap_offset)
                        for p in patches]...)
        active_bitmaps = [p.active_pixel_bitmap for p in patches]
    end

    function pretty_print_galaxy_params(th)
        lnfluxes, unc_pos, ushape = th[1:5], th[6:7], th[8:end]
        pos   = constrain_pos(unc_pos)
        gal_frac_dev, gal_ab, gal_angle, gal_scale = ushape
        @printf "lnfluxes     = %s \n" string(lnfluxes)
        @printf "pos (ra,dec) = %2.4f, %2.4f \n" pos[1] pos[2]
        @printf "gal shape:\n"
        @printf "  frac_dev   = %2.4f \n" gal_frac_dev
        @printf "  ab ratio   = %2.4f \n" gal_ab
        @printf "  angle      = %2.4f \n" gal_angle
        @printf "  scale      = %2.4f \n" gal_scale
    end

    # make galaxy log like function
    function gal_loglike(th::Array{Float64, 1}; print_params=false)

        # unpack location and log fluxes (smushed)
        lnfluxes, unc_pos, ushape = th[1:5], th[6:7], th[8:end]
        pos = unc_pos #constrain_pos(unc_pos)
        gal_frac_dev, gal_ab, gal_angle, gal_scale = ushape

        if print_params
          pretty_print_galaxy_params(th)
        end

        ll = 0.
        for ii in 1:length(imgs)
            img, active_bitmap = imgs[ii], active_bitmaps[ii]

            # sky pixel intensity (sky image)
            background = background_images[ii]

            # create and cache unit flux src image
            src_pixels = zeros(img.H, img.W)
            write_galaxy_unit_flux(pos, img.psf, img.wcs,
                Float64(median(img.nelec_per_nmgy)),
                gal_frac_dev, gal_ab, gal_angle, gal_scale,
                src_pixels;
                offset=offsets[:,ii])

            # image specific flux
            bflux = exp(lnfluxes[img.b])
            if isinf(bflux)
                return -Inf
            end

            # sum per-pixel likelihood contribution
            for h in 1:img.H, w in 1:img.W
                pixel_data = img.pixels[h,w]
                is_active  = active_bitmap[h,w]
                if !isnan(pixel_data) && is_active
                    rate_hw = background[h,w] + bflux*src_pixels[h,w]
                    ll += poisson_lnpdf(pixel_data, rate_hw)
                end
            end
        end

    return ll
    end

    return gal_loglike, constrain_pos, unconstrain_pos
end


##########
# priors #
##########


"""
Create a uniform prior around a RA/DEC location --- specify size of box
by number of pixels
"""
function make_location_prior(img::Image,
                             pos0::Array{Float64, 1};
                             pos_pixel_delta::Array{Float64, 1} = [1., 1.])

    # figure out lower and upper bounds on RA, Dec
    pos0_pix = WCS.world_to_pix(img.wcs, pos0)
    pos0_pix_lower = pos0_pix - .5 * pos_pixel_delta
    pos0_pix_upper = pos0_pix + .5 * pos_pixel_delta
    pos0_world_lower = WCS.pix_to_world(img.wcs, pos0_pix_lower)
    pos0_world_upper = WCS.pix_to_world(img.wcs, pos0_pix_upper)

    # lower and upper bounds on the ra/dec
    ra_lo, ra_hi   = sort([pos0_world_lower[1], pos0_world_upper[1]])
    dec_lo, dec_hi = sort([pos0_world_lower[2], pos0_world_upper[2]])
    @printf " ... limiting RA  to [%2.5f, %2.5f] \n" ra_lo ra_hi
    @printf " ... limiting DEC to [%2.5f, %2.5f] \n" dec_lo dec_hi

    # corresponding uniform log likelihoods
    llra  = log(1./(ra_hi - ra_lo))
    lldec = log(1./(dec_hi - dec_lo))

    function pos_logprior(pos)
        if !inrange(pos[1], ra_lo, ra_hi)
            return -Inf
        end
        if !inrange(pos[2], dec_lo, dec_hi)
            return -Inf
        end
        return llra + lldec
    end

    return pos_logprior, [ra_lo, ra_hi], [dec_lo, dec_hi]
end


function make_gal_logprior()

    # distributions over galaxy parameters
    prior = Model.construct_prior()

    function gal_logprior(th)
        lnfluxes, u, ushape = th[1:5], th[6:7], th[8:end]

        # first compute prior --- make sure all in bounds
        gal_frac_dev, gal_ab, gal_angle, gal_scale = ushape
        if !inrange(gal_frac_dev, 0., 1.)
            return -Inf
        end
        if !inrange(gal_ab, 0., 1.)
            return -Inf
        end
        if !inrange(gal_angle, 0., pi)
            return -Inf
        end
        if !inrange(gal_scale, 1e-5, Inf)
            return -Inf
        end

        # uniform over angle, ll log normal over scale
        llangle = log(pi)
        llscale = logpdf(prior.galaxy.gal_scale, gal_scale)
        if isinf(llangle)
          println(" angle bad!")
        end
        if isinf(llscale)
          println(" scale bad!")
        end
        ll = MCMC.logflux_logprior(lnfluxes; is_star=false) + llangle + llscale
        return ll
    end
    return gal_logprior
end


function make_empty_background_images(imgs::Array{Image})
    background_images = []
    for img in imgs
        # sky pixel intensity (sky image)
        epsilon    = img.sky[1, 1]
        iota       = img.nelec_per_nmgy[1]
        sky_pixels = [epsilon * iota for h=1:img.H, w=1:img.W]
        push!(background_images, sky_pixels)
    end
    return background_images
end


function make_position_transformations(pos0::Array{Float64, 1},
                                       pos_delta::Array{Float64, 1})

    # create a function that constrains the pixel location to be within
    # pos0 +- [pos_delta]
    pos_lo = pos0 .- pos_delta
    pos_hi = pos0 .+ pos_delta
    pos_range = pos_hi .- pos_lo

    function constrain_pos(upos::Array{Float64, 1})
        unit = 1 ./ (1. + exp.(-upos))
        cpos  = (unit.*pos_range) .+ pos_lo
        return cpos
    end

    function unconstrain_pos(cpos::Array{Float64, 1})
        unit = (cpos .- pos_lo) ./ pos_range
        upos = log.(unit) .- log.(1 .- unit)
        return upos
    end

    return constrain_pos, unconstrain_pos
end


#####################################
# simple log prob implementation    #
#####################################

"""
given a single image, create a function that returns log likelihood of image
as a function of the log flux = [lnu, lnr, lng, lni, lnz]
"""
function make_single_image_logflux_loglike(img0::Image,
                                           pos::Array{Float64, 1};
                                           is_star               = true,
                                           gal_frac_dev::Float64 = .5,
                                           gal_ab::Float64       = .7,
                                           gal_angle::Float64    = .79,
                                           gal_scale::Float64    = 4.)
    # sky pixel intensity
    epsilon    = img0.sky[1, 1]
    iota       = img0.nelec_per_nmgy[1]
    sky_pixels = [epsilon * iota for h=1:img0.H, w=1:img0.W]

    # create and cache unit flux src image
    src_pixels = [0. for h=1:img0.H, w=1:img0.W]
    if is_star
        write_star_unit_flux(img0, pos, src_pixels)
    else
        write_galaxy_unit_flux(img0, pos, gal_frac_dev,
                               gal_ab, gal_angle, gal_scale, src_pixels)
    end

    # create log like handle and return
    function lnflux_loglike(lnflux)

        # compute model image
        rates = sky_pixels .+ exp(lnflux)*src_pixels

        # sum per-pixel likelihood contribution
        ll = 0.
        H, W = size(rates)
        for h in 1:H, w in 1:W
            ll += poisson_lnpdf(img0.pixels[h,w], rates[h,w])
        end

        return ll
    end

    return lnflux_loglike
end


"""
create vectorized star log flux log like function with multiple images
"""
function make_logflux_loglike(imgs::Array{Image},
                              pos::Array{Float64, 1};
                              is_star = true,
                              gal_frac_dev::Float64 = .5,
                              gal_ab::Float64       = .7,
                              gal_angle::Float64    = .79,
                              gal_scale::Float64    = 4.)

    # create per image scalar log flux function (caches unit flux pixel image)
    if is_star
        scalar_funs = [make_single_image_logflux_loglike(img, pos;
                          is_star=true)
                       for img in imgs]
    else
        scalar_funs = [make_single_image_logflux_loglike(img, pos;
                          is_star=false,
                          gal_frac_dev=gal_frac_dev,
                          gal_ab=gal_ab,
                          gal_angle=gal_angle,
                          gal_scale=gal_scale)
                       for img in imgs]
    end

    function lnflux_loglike(lnflux::Vector{Float64})
        # for each of those image/function pairs, compute flux-specific ll
        ll = 0.
        for i in 1:length(imgs)
            b   = imgs[i].b
            ll += scalar_funs[i](lnflux[b])
        end
        return ll
    end

    return lnflux_loglike
end


#############################
# Priors and prior samplers #
#############################

const pp = Model.load_prior()

"""
prior over log fluxes (for stars and galaxies)
"""
function logflux_logprior(lnfluxes::Vector{Float64}; is_star::Bool=true)
    # toggle star / galaxy
    type_i = is_star ? 1 : 2

    # convert to color space for prior calc
    lnr, colors = logfluxes_to_colors(lnfluxes)

    # compute brightness distribution
    #llr = logpdf(Normal(pp.r_μ[type_i], 2*pp.r_σ²[type_i]), lnr)
    llr = logpdf(Normal(pp.flux_mean[type_i], 2*pp.flux_var[type_i]), lnr)

    # compute color likelihoods (mixture model --- computes all component lls)
    nc, nk, ns = size(pp.color_mean)
    llk = [logpdf(MvNormal(pp.color_mean[:,k,type_i],
                           pp.color_cov[:,:,k,type_i]), colors)
           for k in 1:nk]
    lnpik = log.(pp.k[:, type_i])
    llc   = Model.logsumexp(llk .+ lnpik)

    # add and return
    return llr + llc
end

"""
More numerically stable converter from log fluxes to colors
"""
function logfluxes_to_colors(lnfluxes::Vector{Float64})
    lnr = lnfluxes[3]
    colors = Vector{Float64}(4)
    colors[1] = lnfluxes[2] - lnfluxes[1]
    colors[2] = lnfluxes[3] - lnfluxes[2]
    colors[3] = lnfluxes[4] - lnfluxes[3]
    colors[4] = lnfluxes[5] - lnfluxes[4]
    return lnr, colors
end


"""
sample fluxes from prior
"""
function sample_logfluxes(; is_star=true, lnr=nothing)
    # toggle star / galaxy
    type_i = is_star ? 1 : 2

    # sample log r
    if lnr == nothing
        lnr = rand(Normal(pp.flux_mean[type_i], 2*pp.flux_var[type_i]))
    end

    # sample colors
    k = rand(Distributions.Categorical(pp.k[:, type_i]))
    c = rand(MvNormal(pp.color_mean[:,k,type_i], pp.color_cov[:,:,k,type_i]))

    # convert to fluxes
    lnfluxes = log.(Model.colors_to_fluxes(lnr, c))
    return lnfluxes
end

"""
sample colors from prior
"""
function sample_colors(; is_star=true)
    # toggle star / galaxy
    type_i = is_star ? 1 : 2
    # sample colors
    k = rand(Distributions.Categorical(pp.k[:, type_i]))
    c = rand(MvNormal(pp.color_mean[:,k,type_i], pp.color_cov[:,:,k,type_i]))
    return c
end

function sample_logr(; is_star=true)
    type_i = is_star ? 1 : 2
    lnr = rand(Normal(pp.flux_mean[type_i], 2*pp.flux_var[type_i]))
    return lnr
end


#function sample_fluxes(i::Int, r_s)
#    k_s = rand(Distributions.Categorical(pp.k[i]))
#    c_s = rand(Distributions.MvNormal(pp.c[i][:, k_s], pp.c[i][:, :, k_s]))
#
#    l_s = Vector{Float64}(5)
#    l_s[3] = r_s
#    l_s[4] = l_s[3] * exp(c_s[3])
#    l_s[5] = l_s[4] * exp(c_s[4])
#    l_s[2] = l_s[3] / exp(c_s[2])
#    l_s[1] = l_s[2] / exp(c_s[1])
#    l_s
#end

"""
Convert catalog entry into unconstrained parameters for star or gal loglikes
"""
function parameters_from_catalog(entry::CatalogEntry;
                                 unconstrain_pos::Function= x-> x,
                                 is_star=true, epsilon=1e-5)
    if is_star
        return vcat([log.(entry.star_fluxes), unconstrain_pos(entry.pos)]...)
    else
        #ushape = Model.unconstrain_gal_shape([
        #  entry.gal_frac_dev, entry.gal_ab, entry.gal_angle, entry.gal_scale])
        ushape = [clamp(entry.gal_frac_dev, epsilon, 1-epsilon),
                  clamp(entry.gal_ab, epsilon, 1-epsilon),
                  clamp(entry.gal_angle, epsilon, pi-epsilon),
                  clamp(entry.gal_scale, epsilon, Inf)]
        return vcat([log.(entry.gal_fluxes), unconstrain_pos(entry.pos), ushape]...)
    end
end
