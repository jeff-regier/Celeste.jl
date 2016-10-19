# Functions to compute the log probability of star parameters and galaxy
# parameters given pixel data
using Distributions
import ..SensitiveFloats: SensitiveFloat, zero_sensitive_float, clear!

#include("../bivariate_normals.jl")
#include("log_prob_util.jl")
EPS = 1e-6

###################################
# log likelihood function makers  #
###################################


"""
Creates a vectorized version of the star logpdf as a function of
unconstrained params.
    star_params = [lnr, lnc1, ..., lnc4, ra, dec]
Args:
  - images: Vector of TiledImage types (data for log_likelihood)
  - active_pixels: Vector of ActivePixels on which the log_likelihood is based
  - S: --- from ElboArgs
  - N:
Returns:
  - star_logpdf  : unnormalized logpdf function handle that takes in a flat,
                   unconstrained array as parameter
  - star_logprior: star param log prior function handle that takes in same
                   flat, unconstrained array as parameter
"""
function make_star_logpdf(images::Vector{TiledImage},
                          active_pixels::Vector{ActivePixel},
                          S::Int64,
                          N::Int64,
                          source_params::Vector{Vector{Float64}},
                          tile_source_map::Vector{Matrix{Vector{Int}}},
                          patches::Matrix{SkyPatch},
                          active_sources::Vector{Int},
                          psf_K::Int64,
                          num_allowed_sd::Float64)

    # define star prior log probability density function
    prior    = construct_prior()
    subprior = prior.star

    function star_logprior(state::Vector{Float64})
        brightness, colors, u = state[1], state[2:5], state[6:end]
        return color_logprior(brightness, colors, prior, true)
    end

    # define star log joint probability density function
    function star_logpdf(state::Vector{Float64})
        ll_prior = star_logprior(state)
        brightness, colors, position = state[1], state[2:5], state[6:end]
        dummy_gal_shape = [.1, .1, .1, .1]
        ll_like  = state_log_likelihood(true, brightness, colors, position,
                                        dummy_gal_shape, images,
                                        active_pixels,
                                        patches,
                                        active_sources,
                                        psf_K, num_allowed_sd,
                                        source_params,
                                        tile_source_map, S, N)
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
  - images: Vector of TiledImage types (data for log_likelihood)
  - active_pixels: Vector of ActivePixels on which the log_likelihood is based
  - ea: ElboArgs book keeping argument
Returns:
  - gal_logpdf  : unnormalized logpdf function handle that takes in a flat,
                 unconstrained array as parameter
  - gal_logprior: star param log prior function handle that takes in same
                 flat, unconstrained array as parameter
"""
function make_galaxy_logpdf(images::Vector{TiledImage},
                            active_pixels::Vector{ActivePixel},
                            S::Int64,
                            N::Int64,
                            source_params::Vector{GalaxyPosParams}, #vp::VariationalParams{Float64},
                            tile_source_map::Vector{Matrix{Vector{Int}}},
                            patches::Matrix{SkyPatch},
                            active_sources::Vector{Int},
                            psf_K::Int64,
                            num_allowed_sd::Float64)

    # define star prior log probability density function
    prior    = construct_prior()
    subprior = prior.galaxy

    function galaxy_logprior(state::Vector{Float64})
        brightness, colors, u, gal_shape =
            state[1], state[2:5], state[6:7], state[8:end]

        # brightness prior
        ll_b = color_logprior(brightness, colors, prior, true)
        ll_s = shape_logprior(constrain_gal_shape(gal_shape), prior)
        return ll_b + ll_s
    end

    # define star log joint probability density function
    function galaxy_logpdf(state::Vector{Float64})
        ll_prior = galaxy_logprior(state)

        brightness, colors, position, gal_shape =
            state[1], state[2:5], state[6:7], state[8:end]

        ll_like  = state_log_likelihood(false, brightness, colors, position,
                                        constrain_gal_shape(gal_shape), images,
                                        active_pixels,
                                        patches,
                                        active_sources, psf_K, num_allowed_sd,
                                        source_params,
                                        tile_source_map, S, N)
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
  - images: vector of TiledImage types (data for likelihood)
  - active_pixels: vector of ActivePixels over which the ll is summed
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
                              images::Vector{TiledImage},   # list of images with source pixel data
                              active_pixels::Vector{ActivePixel}, # which pixels enter into the likelihood
                              patches::Matrix{SkyPatch},    # formerly of ElboArgs
                              active_sources::Vector{Int},  # formerly of ElboArgs
                              psf_K::Int64,                 # number of PSF Comps
                              num_allowed_sd::Float64,      # ...
                              source_params::Vector{Vector{Float64}}, # list of background sources
                              tile_source_map::Vector{Matrix{Vector{Int}}},
                              S::Int64,
                              N::Int64)
    # TODO: cache the background rate image!! --- does not need to be recomputed at each ll eval
    # convert brightness/colors to fluxes for scaling
    fluxes = colors_to_fluxes(brightness, colors)

    # create objects needed to compute the mean poisson value per pixel
    # (similar to ElboDeriv.process_active_pixels!)
    model_vars =
      ModelIntermediateVariables(Float64, S, length(active_sources))
    model_vars.calculate_derivs = false
    model_vars.calculate_hessian= false

    # load star/gal mixture components (make sure these reflect
    gdev, gaxis, gangle, gscale = gal_shape
    source_params[1][lidx.u]       = position
    source_params[1][lidx.e_dev]   = gdev
    source_params[1][lidx.e_axis]  = gaxis
    source_params[1][lidx.e_angle] = gangle
    source_params[1][lidx.e_scale] = gscale
    star_mcs_vec = Array(Array{BvnComponent{Float64}, 2}, N)
    gal_mcs_vec  = Array(Array{GalaxyCacheComponent{Float64}, 4}, N)
    for b=1:N
        star_mcs_vec[b], gal_mcs_vec[b] =
            load_bvn_mixtures(S, patches, source_params, active_sources, psf_K, b,
                              calculate_derivs=model_vars.calculate_derivs,
                              calculate_hessian=model_vars.calculate_hessian)
    end

    # iterate over the pixels, summing pixel-specific poisson rates
    ll = 0.
    for pixel in active_pixels
        tile         = images[pixel.n].tiles[pixel.tile_ind]
        tile_sources = tile_source_map[pixel.n][pixel.tile_ind]
        this_pixel   = tile.pixels[pixel.h, pixel.w]
        pixel_band   = tile.b

        # compute the unit-flux pixel values
        populate_fsm_vecs!(model_vars, patches, active_sources,
                           num_allowed_sd,
                           tile_sources, tile,
                           pixel.h, pixel.w,
                           gal_mcs_vec[pixel.n], star_mcs_vec[pixel.n])

        # compute the background rate for this pixel
        background_rate    = tile.epsilon_mat[pixel.h, pixel.w]
        background_sources = tile_sources[tile_sources.!=1]
        for s in background_sources
            println("background s: ", s)
            #TODO: compute background rate conditional on source_params
            #flux_s = variational_params_to_fluxes(s, vp)
            #rate_s = is_star ? model_vars.fs0m_vec[s].v : model_vars.fs1m_vec[s].v
            #rate_s *= flux_s[pixel_band]
            #background_rate += rate_s
        end

        # this source's rate, add to background for total
        this_rate  = is_star ? model_vars.fs0m_vec[1].v : model_vars.fs1m_vec[1].v
        pixel_rate = fluxes[pixel_band]*this_rate + background_rate

        # multiply by image's gain for this pixel
        rate     = pixel_rate * tile.iota_vec[pixel.h]
        pixel_ll = logpdf(Poisson(rate[1]), round(Int, this_pixel))
        ll += pixel_ll
    end
    return ll
end


###########################################################################
# priors - for computing prior log probs of star/gal parameters           #
###########################################################################


type StarPrior
    brightness::LogNormal
    color_component::Categorical
    colors::Vector{MvNormal}
end


type GalaxyPrior
    brightness::LogNormal
    color_component::Categorical
    colors::Vector{MvNormal}
    gal_scale::LogNormal
    gal_ab::Beta
    gal_fracdev::Beta
end


type Prior
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
        LogNormal(pp.r_mean[1], pp.r_var[1]),
        Categorical(pp.k[:,1]),
        [MvNormal(pp.c_mean[:,k,1], pp.c_cov[:,:,k,1]) for k in 1:2])

    gal_prior = GalaxyPrior(
        LogNormal(pp.r_mean[2], pp.r_var[2]),
        Categorical(pp.k[:,2]),
        [MvNormal(pp.c_mean[:,k,2], pp.c_cov[:,:,k,2]) for k in 1:2],
        LogNormal(0, 10),
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
    ll_component  = [logpdf(subprior.colors[k], colors) for k in 1:2]
    ll_color      = logsumexp(ll_component + log(subprior.color_component.p))
    return ll_brightness + ll_color
end


function shape_logprior(gal_shape::Vector{Float64}, prior::Prior)
    # position and gal_angle have uniform priors--we ignore them
    gdev, gaxis, gangle, gscale = gal_shape
    ll_shape = 0.
    ll_shape += logpdf(prior.galaxy.gal_scale, gscale)
    ll_shape += logpdf(prior.galaxy.gal_ab, gaxis)
    ll_shape += logpdf(prior.galaxy.gal_fracdev, gdev)
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
    colors = Array(Float64, 4)
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
    ret    = Array(Float64, 5)
    lnr    = brightness
    ret[3] = lnr     # r flux
    ret[4] = lnr + colors[3]        # ln(i/r) = c3 => lni = lnr - c3
    ret[5] = ret[4] + colors[4]     # ln(z/i) = c4 => lnz = lni - c4
    ret[2] = -colors[2] + lnr       # ln(r/g) = c2 => lng = c2 + lnr
    ret[1] = -colors[1] + ret[2]    # ln(g/u) = c1 => lnu = c1 + lng
    return exp(ret)
end


#######################################################################
# galaxy shape transformation (unconstrained => constrained and back) #
#######################################################################


function constrain_gal_shape(unc_gal_shape::Vector{Float64})
    gdev, gaxis, gangle, gscale = unc_gal_shape
    constr_shape    = Array(Float64, 4)
    constr_shape[1] = clamp(sigmoid(gdev), EPS, 1-EPS)
    constr_shape[2] = clamp(sigmoid(gaxis), EPS, 1-EPS)
    constr_shape[3] = gangle       # TODO put this between [0, 2pi]
    constr_shape[4] = clamp(exp(gscale), EPS, Inf)
    return constr_shape
end


function unconstrain_gal_shape(con_gal_shape::Vector{Float64})
    gdev, gaxis, gangle, gscale = con_gal_shape
    unc_shape    = Array(Float64, 4)
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
                    clamp(entry.gal_frac_dev, EPS, 1.-EPS),
                    clamp(entry.gal_ab, EPS, 1.-EPS),
                    entry.gal_angle,
                    clamp(entry.gal_scale, EPS, Inf)
                    ])
    param_vec = [brightness; colors; entry.pos; gal_shape]
    return param_vec
end


function elbo_args_vp_to_star_state(vp::Array{Float64,1})
    fluxes      = variational_params_to_fluxes(vp, 1)
    lnr, colors = fluxes_to_colors(fluxes)
    ra, dec     = vp[ids.u[1]], vp[ids.u[2]]
    return [[lnr]; colors; [ra, dec]]
end


function elbo_args_vp_to_galaxy_state(vp::Array{Float64, 1})
    shared_params = elbo_args_vp_to_star_state(vp, 2)
    gal_shape = [vp[ids.e_dev]  ,
                 vp[ids.e_axis] ,
                 vp[ids.e_angle],
                 vp[ids.e_scale]]
    return [shared_params; unconstrain_gal_shape(gal_shape)]
end


function variational_params_to_latent_state_params(vp::Array{Float64, 1})
    # create a float array of the appropriate length
    ret = Array(Float64, length(lidx))

    # galaxy shape params
    ret[lidx.u]       = vp[ids.u]
    ret[lidx.e_dev]   = vp[ids.e_dev]
    ret[lidx.e_axis]  = vp[ids.e_axis]
    ret[lidx.e_scale] = vp[ids.e_scale]

    # star, gal r flux
    star_lnr, star_cols =
        fluxes_to_colors(variational_params_to_fluxes(vp, 1))
    gal_lnr, gal_cols =
        fluxes_to_colors(variational_params_to_fluxes(vp, 2))
    ret[lidx.r] = [star_lnr, gal_lnr]
    ret[lidx.c] = hcat([star_cols, gal_cols]...)

    # set the prob star/prob gal
    ret[lidx.a] = vp[ids.a]

    ret
end


function catalog_entry_to_latent_state_params(ce::CatalogEntry)
    # create a float array of the appropriate length
    ret = Array(Float64, length(lidx))

    # galaxy shape params
    ret[lidx.u]       = ce.pos
    ret[lidx.e_dev]   = ce.gal_frac_dev
    ret[lidx.e_axis]  = ce.gal_ab
    ret[lidx.e_scale] = ce.gal_scale

    # star, gal r flux
    star_lnr, star_cols = fluxes_to_colors(ce.star_fluxes)
    gal_lnr, gal_cols   = fluxes_to_colors(ce.gal_fluxes)
    ret[lidx.r]         = [star_lnr, gal_lnr]
    ret[lidx.c]         = hcat([star_cols, gal_cols]...)

    # set the prob star/prob gal
    ret[lidx.a] = clamp(ce.is_star, EPS, 1-EPS)
    ret
end


function variational_params_to_fluxes(vs::Array{Float64,1}, i::Int)
    ret = Array(Float64, 5)
    ret[3] = exp(vs[ids.r1[i]] + 0.5 * vs[ids.r2[i]])
    ret[4] = ret[3] * exp(vs[ids.c1[3, i]])
    ret[5] = ret[4] * exp(vs[ids.c1[4, i]])
    ret[2] = ret[3] / exp(vs[ids.c1[2, i]])
    ret[1] = ret[2] / exp(vs[ids.c1[1, i]])
    ret
end


###################
# other util funs #
###################

#TODO is there a better place for this generic function? --- acm
function logsumexp(a::Vector{Float64})
    a_max = maximum(a)
    out = log(sum(exp(a - a_max)))
    out += a_max
    return out
end


function logit(a::Float64)
    return log(a) - log(1.0-a)
end


function sigmoid(a::Float64)
    return 1. / (1. + exp(-a))
end
