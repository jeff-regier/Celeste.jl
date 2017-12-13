# Functions to compute the log probability of star parameters and galaxy
# parameters given pixel data
using Distributions
import ..SensitiveFloats: SensitiveFloat
import JLD

const eps_prob_a = 1e-6

###################################
# log likelihood function makers  #
###################################


"""
Creates a vectorized version of the star logpdf as a function of
unconstrained params.
    star_params = [lnr, lnc1, ..., lnc4, ra, dec]
Args:
  - images: Vector of Image types (data for log_likelihood)
  - S: --- from ElboArgs
  - N:
Returns:
  - star_logpdf  : unnormalized logpdf function handle that takes in a flat,
                   unconstrained array as parameter
  - star_logprior: star param log prior function handle that takes in same
                   flat, unconstrained array as parameter
"""
function make_star_logpdf(images::Vector{<:Image},
                          S::Int64,
                          N::Int64,
                          source_params::Vector{Vector{Float64}},
                          patches::Matrix{ImagePatch},
                          active_sources::Vector{Int},
                          psf_K::Int64)
    # define star prior log probability density function
    prior = construct_prior()
    subprior = prior.star

    function star_logprior(state::Vector{Float64})
        brightness, colors, pos = state[1], state[2:5], state[6:end]
        return color_logprior(brightness, colors, prior, true)
    end

    # define star log joint probability density function
    function star_logpdf(state::Vector{Float64})
        ll_prior = star_logprior(state)
        brightness, colors, position = state[1], state[2:5], state[6:end]
        dummy_gal_shape = [.1, .1, .1, .1]
        ll_like  = state_log_likelihood(true, brightness, colors, position,
                                        dummy_gal_shape, images,
                                        patches,
                                        active_sources,
                                        psf_K,
                                        source_params,
                                        S, N)
        return ll_like + ll_prior
    end

    return star_logpdf, star_logprior
end


"""
Creates a vectorized version of a galaxy logpdf as a function of
unconstrained params
    gal_params = [lnr, lnc1, ..., lnc4, ra, dec, shp1, shp2, shp3, shp4]
where the flux and position params are the same as the star case, and the
shape params are unconstrained versions of (gdev, gaxis, gangle, gscale)
Args:
  - images: Vector of Image types (data for log_likelihood)
  - ea: ElboArgs book keeping argument
Returns:
  - gal_logpdf  : unnormalized logpdf function handle that takes in a flat,
                 unconstrained array as parameter
  - gal_logprior: star param log prior function handle that takes in same
                 flat, unconstrained array as parameter
"""
function make_galaxy_logpdf(images::Vector{<:Image},
                            S::Int64,
                            N::Int64,
                            source_params::Vector{Vector{Float64}},
                            patches::Matrix{ImagePatch},
                            active_sources::Vector{Int},
                            psf_K::Int64)
    # define galaxy prior function
    prior    = construct_prior()
    subprior = prior.galaxy

    function galaxy_logprior(state::Vector{Float64})
        brightness, colors, pos, gal_shape =
            state[1], state[2:5], state[6:7], state[8:end]

        # brightness prior
        ll_b = color_logprior(brightness, colors, prior, true)
        ll_s = shape_logprior(constrain_gal_shape(gal_shape), prior)
        return ll_b + ll_s
    end

    # define galaxy log joint probability density function
    function galaxy_logpdf(state::Vector{Float64})
        ll_prior = galaxy_logprior(state)
        brightness, colors, position, gal_shape =
            state[1], state[2:5], state[6:7], state[8:end]
        ll_like  = state_log_likelihood(false, brightness, colors, position,
                                        constrain_gal_shape(gal_shape), images,
                                        patches,
                                        active_sources, psf_K,
                                        source_params,
                                        S, N)
        return ll_like + ll_prior
    end

    return galaxy_logpdf, galaxy_logprior
end


"""
Log likelihood of a single source given source params.
Args:
  - is_star: bool describing the type of source
  - brightness: log r-band value
  - colors: array of colors
  - position: ra/dec of source
  - gal_shape: vector of galaxy shape params (used if is_star=false)
  - images: vector of Image types (data for likelihood)
  - ea: ElboArgs object that maintains params for all sources
Returns:
  - result: a scalar describing the log likelihood of the
            (brightness,colors,position,gal_shape) params conditioned on
            the rest of the args
"""
function state_log_likelihood(is_star::Bool,                # source is star
                              brightness::Float64,          # source log r brightness
                              colors::Vector{Float64},      # source vector of colors
                              position::Vector{Float64},    # source position
                              gal_shape::Vector{Float64},   # source gal shape
                              images::Vector{<:Image},   # list of images with source pixel data
                              patches::Matrix{ImagePatch},    # formerly of ElboArgs
                              active_sources::Vector{Int},  # formerly of ElboArgs
                              psf_K::Int64,                 # number of PSF Comps
                              source_params::Vector{Vector{Float64}}, # list of background sources
                              S::Int64,
                              N::Int64)
    # TODO: cache the background rate image!! --- does not need to be recomputed at each ll eval
    # convert brightness/colors to fluxes for scaling
    fluxes = colors_to_fluxes(brightness, colors)

    # create objects needed to compute the mean poisson value per pixel
    # (similar to ElboDeriv.process_active_pixels!)
    bvn_derivs = BivariateNormalDerivatives{Float64}()
    fs0m = SensitiveFloat{Float64}(length(StarPosParams), 1, false, false)
    fs1m = SensitiveFloat{Float64}(length(GalaxyPosParams), 1, false, false)

    # load star/gal mixture components (make sure these reflect
    gdev, gaxis, gangle, gscale = gal_shape
    source_params[1][lidx.pos]       = position
    source_params[1][lidx.gal_frac_dev]   = gdev
    source_params[1][lidx.gal_axis_ratio]  = gaxis
    source_params[1][lidx.gal_angle] = gangle
    source_params[1][lidx.gal_radius_px] = gscale

    # iterate over the pixels, summing pixel-specific poisson rates
    ll = 0.

    @assert length(active_sources) == 1

    for n in 1:N
        img = images[n]

        star_mcs, gal_mcs = load_bvn_mixtures(S, patches,
                              source_params, active_sources, psf_K, n,
                              calculate_gradient=false,
                              calculate_hessian=false)

        p = patches[active_sources[1], n]
        H2, W2 = size(p.active_pixel_bitmap)
        for w2 in 1:W2, h2 in 1:H2
            # (h2, w2) index the local patch, while (h, w) index the image
            h = p.bitmap_offset[1] + h2
            w = p.bitmap_offset[2] + w2

            if !p.active_pixel_bitmap[h2, w2]
                continue
            end

            # compute the unit-flux pixel values
            pixel_rate = img.sky[h, w]
            for s in 1:S
                params_s  = source_params[s]

                star_light_density!(fs0m, p, h, w, params_s[lidx.pos], false)
                populate_gal_fsm!(fs1m, bvn_derivs, s, h, w, false, p.wcs_jacobian, gal_mcs)

                if s == 1
                    this_rate  = is_star ? fs0m.v[] : fs1m.v[]
                    pixel_rate += fluxes[img.b] * this_rate
                else
                    # determine if background source is star/gal; get fluxes
                    s_is_star = params_s[lidx.is_star[1,1]] > .5
                    type_idx  = s_is_star ? 1 : 2
                    flux_s    = colors_to_fluxes(params_s[lidx.flux[type_idx]],
                                                 params_s[lidx.color[:,type_idx]])
                    rate_s = s_is_star ? fs0m.v[] : fs1m.v[]
                    rate_s *= flux_s[img.b]
                    pixel_rate += rate_s
                end
            end

            # multiply by image's gain for this pixel
            rate = pixel_rate * img.nelec_per_nmgy[h]
            pixel_ll = logpdf(Poisson(rate[1]), round(Int, img.pixels[h, w]))
            ll += pixel_ll
        end
    end

    return ll
end


###########################################################################
# priors - for computing prior log probs of star/gal parameters           #
###########################################################################


struct StarPrior
    brightness::LogNormal
    color_component::Categorical
    colors::Vector{MvNormal}
end


struct GalaxyPrior
    brightness::LogNormal
    color_component::Categorical
    colors::Vector{MvNormal}
    gal_radius_px::LogNormal
    gal_axis_ratio::Beta
    gal_frac_dev::Beta
end


struct Prior
    is_star::Bernoulli
    star::StarPrior
    galaxy::GalaxyPrior
end


"""
Construct a `Prior` object, contains StarPrior and GalaxyPrior objects
"""
function construct_prior()
    pp = load_prior()
    star_prior = StarPrior(
        LogNormal(pp.flux_mean[1], sqrt(pp.flux_var[1])),
        Categorical(pp.k[:,1]),
        [MvNormal(pp.color_mean[:,k,1], pp.color_cov[:,:,k,1]) for k in 1:NUM_COLOR_COMPONENTS])

    gal_prior = GalaxyPrior(
        LogNormal(pp.flux_mean[2], sqrt(pp.flux_var[2])),
        Categorical(pp.k[:,2]),
        [MvNormal(pp.color_mean[:,k,2], pp.color_cov[:,:,k,2]) for k in 1:NUM_COLOR_COMPONENTS],
        LogNormal(pp.gal_radius_px_mean, sqrt(pp.gal_radius_px_var)),
        Beta(1, 1),
        Beta(1, 1))

    return Prior(Bernoulli(.5), star_prior, gal_prior)
end


"""
Return the log prior probability of a (brightness,color) pair based on
prior probability objects constructed by `construct_prior()`
"""
function color_logprior(brightness::Float64,
                        colors::Vector{Float64},
                        prior::Prior,
                        is_star::Bool)
    subprior      = is_star ? prior.star : prior.galaxy
    ll_brightness = logpdf(subprior.brightness, exp(brightness))
    ll_component  = [logpdf(subprior.colors[k], colors) for k in 1:NUM_COLOR_COMPONENTS]
    ll_color      = logsumexp(ll_component + log.(subprior.color_component.p))
    return ll_brightness + ll_color
end


function shape_logprior(gal_shape::Vector{Float64}, prior::Prior)
    # position and gal_angle have uniform priors--we ignore them
    gdev, gaxis, gangle, gscale = gal_shape
    ll_shape = 0.
    ll_shape += logpdf(prior.galaxy.gal_radius_px, gscale)
    ll_shape += logpdf(prior.galaxy.gal_axis_ratio, gaxis)
    ll_shape += logpdf(prior.galaxy.gal_frac_dev, gdev)
    return ll_shape
end


#################################################################
# transformation of fluxes = (u, g, r, i, z) in nanomaggies to  #
# (bright,color) pairs                                          #
#################################################################


"""
Translate between fluxes (nmgy rates) and colors (on which
LogNormal priors are placed)
"""
function fluxes_to_colors(fluxes::Vector{Float64})
    lnr    = log(fluxes[3])
    colors = Vector{Float64}(4)
    colors[1] = log(fluxes[2]) - log(fluxes[1])
    colors[2] = log(fluxes[3]) - log(fluxes[2])
    colors[3] = log(fluxes[4]) - log(fluxes[3])
    colors[4] = log(fluxes[5]) - log(fluxes[4])
    return lnr, colors
end


"""
Translate from the (brightness, color) parameterization to nmgy fluxes
"""
function colors_to_fluxes(brightness::Float64, colors::Vector{Float64})
    # build up log fluxes
    ret    = Vector{Float64}(5)
    lnr    = brightness
    ret[3] = lnr     # r flux
    ret[4] = lnr + colors[3]        # ln(i/r) = c3 => lni = lnr - c3
    ret[5] = ret[4] + colors[4]     # ln(z/i) = c4 => lnz = lni - c4
    ret[2] = -colors[2] + lnr       # ln(r/g) = color_var => lng = color_var + lnr
    ret[1] = -colors[1] + ret[2]    # ln(g/u) = color_mean => lnu = color_mean + lng
    return exp.(ret)
end


#######################################################################
# galaxy shape transformation (unconstrained => constrained and back) #
#######################################################################


function constrain_gal_shape(unc_gal_shape::Vector{Float64})
    gdev, gaxis, gangle, gscale = unc_gal_shape
    constr_shape    = Vector{Float64}(4)
    constr_shape[1] = clamp(sigmoid(gdev), eps_prob_a, 1-eps_prob_a)
    constr_shape[2] = clamp(sigmoid(gaxis), eps_prob_a, 1-eps_prob_a)
    constr_shape[3] = gangle       # TODO put this between [0, 2pi]
    constr_shape[4] = clamp(exp(gscale), eps_prob_a, Inf)
    return constr_shape
end


function unconstrain_gal_shape(con_gal_shape::Vector{Float64})
    gdev, gaxis, gangle, gscale = con_gal_shape
    unc_shape    = Vector{Float64}(4)
    unc_shape[1] = logit(gdev)
    unc_shape[2] = logit(gaxis)
    unc_shape[3] = gangle
    unc_shape[4] = log(gscale)
    return unc_shape
end


######################################################
# initialize from catalog and variational parameters #
######################################################


function init_star_state(entry::CatalogEntry)
    brightness, colors = fluxes_to_colors(entry.star_fluxes)
    param_vec = [brightness; colors; entry.pos]
end


function init_galaxy_state(entry::CatalogEntry)
    brightness, colors = fluxes_to_colors(entry.gal_fluxes)
    #gdev, gaxis, gangle, gscale = gal_shape
    gal_shape = unconstrain_gal_shape([
                    clamp(entry.gal_frac_dev, eps_prob_a, 1.-eps_prob_a),
                    clamp(entry.gal_axis_ratio, eps_prob_a, 1.-eps_prob_a),
                    entry.gal_angle,
                    clamp(entry.gal_radius_px, eps_prob_a, Inf)
                    ])
    param_vec = [brightness; colors; entry.pos; gal_shape]
    return param_vec
end


function catalog_entry_to_latent_state_params(ce::CatalogEntry)
    # create a float array of the appropriate length
    ret = Vector{Float64}(length(lidx))

    # galaxy shape params
    ret[lidx.pos]       = ce.pos
    ret[lidx.gal_frac_dev]   = ce.gal_frac_dev
    ret[lidx.gal_axis_ratio]  = ce.gal_axis_ratio
    ret[lidx.gal_radius_px] = ce.gal_radius_px
    ret[lidx.gal_angle] = ce.gal_angle

    # star, gal r flux
    star_lnr, star_cols = fluxes_to_colors(ce.star_fluxes)
    gal_lnr, gal_cols   = fluxes_to_colors(ce.gal_fluxes)
    ret[lidx.flux]         = [star_lnr, gal_lnr]
    ret[lidx.color]         = hcat([star_cols, gal_cols]...)

    # set the prob star/prob gal
    ret[lidx.is_star] = clamp(ce.is_star, eps_prob_a, 1-eps_prob_a)
    ret
end


function extract_star_state(ls::Array{Float64, 1})
    return [[ls[lidx.flux[1]]]; ls[lidx.color[:, 1]]; ls[lidx.pos]]
end


function extract_galaxy_state(ls::Array{Float64, 1})
    gal_shape = [ls[lidx.gal_frac_dev]  ,
                 ls[lidx.gal_axis_ratio] ,
                 ls[lidx.gal_angle],
                 ls[lidx.gal_radius_px]]
    return [[ls[lidx.flux[2]]]; ls[lidx.color[:, 2]]; ls[lidx.pos];
            unconstrain_gal_shape(gal_shape)]
end


###################
# other util funs #
###################

#TODO is there a better place for this generic function? --- acm
function logsumexp(a::Vector{Float64})
    a_max = maximum(a)
    out = log(sum(exp, a - a_max))
    return out + a_max
end


function logit(a::Float64)
    return log(a) - log(1.0-a)
end


function sigmoid(a::Float64)
    return 1. / (1. + exp(-a))
end
