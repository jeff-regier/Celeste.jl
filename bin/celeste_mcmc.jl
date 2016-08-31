#!/usr/bin/env julia

# ./celeste.jl infer-box 200 200.5 38.1 38.35

using Distributions

using Celeste
import Celeste.Model: PsfComponent, psf_K, galaxy_prototypes, D, Ia, prior
using Celeste: Model, ElboDeriv, Infer
import Celeste: WCSUtils, PSF, RunCamcolField, load_images
import Celeste.ElboDeriv: ActivePixel, BvnComponent, GalaxyCacheComponent

const MIN_FLUX = 2.0

type LatentState
    is_star::Bool
    brightness::Float64
    color_component::Int64
    colors::Vector{Float64}
    position::Vector{Float64}
    gal_scale::Float64
    gal_angle::Float64
    gal_ab::Float64
    gal_fracdev::Float64
end

type ModelParams
    sky_intensity::Float64
    nmgy_to_photons::Float64
    psf::Vector{PsfComponent}
    entry::CatalogEntry
    neighbors::Vector{CatalogEntry}
end


type StarPrior
    brightness::LogNormal
    color_component::Categorical
    colors::Vector{MvLogNormal}
end


type GalaxyPrior
    brightness::LogNormal
    color_component::Categorical
    colors::Vector{MvLogNormal}
    gal_scale::LogNormal
    gal_ab::Beta
    gal_fracdev::Beta
end


type Prior
    is_star::Bernoulli
    star::StarPrior
    galaxy::GalaxyPrior
end


type SampleResult
    star_samples::Vector{StarState}
    galaxy_samples::Vector{GalaxyState}
    type_samples::Vector{Int}
    star_lls::Vector{Float64}
    galaxy_lls::Vector{Float64}
    type_lls::Vector{Float64}
end


"""
Run mcmc sampler for a particular catalog entry given neighbors.  First runs
star-only sampler, then run gal-only sampler, and finally combines the two
chains.

Args:
  - entry: a CatalogEntry corresponding to the source being inferred
  - neighbors: a vector of CatalogEntry objects nearby 'entry'

Returns:
  - result: a SampleResult object that contains a vector of StarState
            and GalaxyState parameters, among other loglikelihood vectors

"""
function run_single_source_sampler(entry::CatalogEntry,
                                   neighbors::Vector{CatalogEntry}, 
                                   images::Vector{TiledImage})
    # preprocssing
    cat_local = vcat(entry, neighbors)
    vp = Vector{Float64}[init_source(ce) for ce in cat_local]
    patches, tile_source_map = Infer.get_tile_source_map(images, cat_local)
    ea = ElboDeriv.ElboArgs(images, vp, tile_source_map, patches, [1])
    Infer.fit_object_psfs!(ea, ea.active_sources)
    Infer.trim_source_tiles!(ea)
    active_pixels = ElboDeriv.get_active_pixels(ea)

    # generate the star logpdf
    star_logpdf  = make_star_logpdf(images, active_pixels, ea)
    star_state = [.1 for i in 1:7]
    println("Star logpdf: ", star_logpdf(star_state))

    #star_state   = init_star_state()
    #star_samples, star_lls = run_slice_sampler(star_logpdf, star_state)

    # generate the galaxy logpdf and sample
    #gal_state   = init_gal_state()
    #gal_logpdf  = make_gal_logpdf(images, active_pixels, ea, prior)
    #gal_samples, gal_lls  = run_slice_sampler(gal_logpdf, gal_state)

    # generate pointers to star/gal type (to infer p(star | data))
    #type_samples, type_lls = run_star_gal_switcher(star_lls, gal_lls)

    # return all samples
    #return SampleResult(star_samples, gal_samples, type_samples,
    #                    star_lls, gal_lls, type_lls)
end



"""
Creates a vectorized version of the star logpdf
"""
function make_star_logpdf(images::Vector{TiledImage},
                          active_pixels::Vector{ActivePixel},
                          ea::ElboArgs)

    # define star prior log probability density function
    prior    = load_prior()
    subprior = prior.star

    function star_log_prior(state::Vector{Float64})
        brightness, colors, u = state[1], state[2:5], state[6:end]
        ll_brightness = logpdf(subprior.brightness, brightness)
        ll_component  = [logpdf(subprior.colors[k], colors) for k in 1:2]
        ll_color      = logsumexp(ll_component + log(subprior.color_component.p))
        return ll_brightness + ll_color
    end

    # define star log joint probability density function
    function star_logpdf(state::Vector{Float64})
        ll_prior = star_log_prior(state)

        brightness, colors, position = state[1], state[2:5], state[6:end]
        state = LatentState(true,  #is_star,
                            brightness,
                            1, # color_component (not needed)
                            colors,
                            position,
                            .1, .1, .1, .1)
        ll_like  = state_log_likelihood([state],
                                        images, active_pixels, ea)
        return ll_like + ll_prior
    end

    return star_logpdf
end


function state_log_prior(states::Vector{LatentState},
                         prior::Prior)
    # prior
    log_prior = 0.0
    for state in states
        #state = states[s]
        log_prior += logpdf(prior.is_star, state.is_star ? 1 : 0)

        subprior = state.is_star ? prior.star : prior.galaxy
        log_prior += logpdf(subprior.brightness, state.brightness)
        log_prior += logpdf(subprior.color_component, state.color_component)
        log_prior += logpdf(subprior.colors[state.color_component], state.colors)

        # position and gal_angle have uniform priors--we ignore them

        if !state.is_star
            log_prior += logpdf(subprior.gal_scale, state.gal_scale)
            log_prior += logpdf(subprior.gal_ab, state.gal_ab)
            log_prior += logpdf(subprior.gal_fracdev, state.gal_fracdev)
        end
    end
    return log_prior
end


function state_log_likelihood(is_star::Bool,
                              brightness::Float64,
                              colors::Vector{Float64},
                              position::Vector{Float64},
                              gal_shape::Vector{Float64},
                              images::Vector{TiledImage},
                              active_pixels::Vector{ActivePixel},
                              ea::ElboArgs)

    # convert brightness/colors to fluxes for scaling
    fluxes = brightness_to_fluxes(brightness, colors)

    # make sure elbo-args reflects the position and galaxy shape passed in
    ea.vp[1][ids.u[1]] = position[1]
    ea.vp[1][ids.u[2]] = position[2]
    ea.vp[1][ids.e_dev]   = gal_shape[1]
    ea.vp[1][ids.e_axis]  = gal_shape[2]
    ea.vp[1][ids.e_angle] = gal_shape[3]
    ea.vp[1][ids.e_scale] = gal_shape[4]

    # create objects needed to compute the mean poisson value per pixel
    # (similar to ElboDeriv.process_active_pixels!)
    elbo_vars =
      ElboDeriv.ElboIntermediateVariables(Float64, ea.S, length(ea.active_sources))
    elbo_vars.calculate_derivs = false
    elbo_vars.calculate_hessian= false

    #TODO: load `star_mcs` and `gal_mcs`, as shown in `ElboDeriv.process_active_pixels!`
    star_mcs_vec = Array(Array{BvnComponent{Float64}, 2}, ea.N)
    gal_mcs_vec  = Array(Array{GalaxyCacheComponent{Float64}, 4}, ea.N)
    for b=1:ea.N
        star_mcs_vec[b], gal_mcs_vec[b] =
            ElboDeriv.load_bvn_mixtures(ea, b,
                calculate_derivs=elbo_vars.calculate_derivs,
                calculate_hessian=elbo_vars.calculate_hessian)
    end

    # iterate over the pixels, summing pixel-wise poisson rates
    for pixel in active_pixels
        tile = ea.images[pixel.n].tiles[pixel.tile_ind]
        band = ea.images[pixel.n].band
        tile_sources = ea.tile_source_map[pixel.n][pixel.tile_ind]
        this_pixel = tile.pixels[pixel.h, pixel.w]

        # set the position and galaxy shape parameters
        ea.vp[1][ids.u[1]] = state.position[1]
        ea.vp[1][ids.u[2]] = state.position[2]

        ElboDeriv.populate_fsm_vecs!(elbo_vars, ea, tile_sources, tile,
                                     pixel.h, pixel.w,
                                     gal_mcs_vec[pixel.n], star_mcs_vec[pixel.n])

        rate = tile.epsilon_mat[pixel.h, pixel.w]
        for s in tile_sources
            state = states[s]
            rate += state.is_star ? elbo_vars.fs0m_vec[s].v : elbo_vars.fs1m_vec[s].v
        end
        rate *= tile.iota_vec[pixel.h]

        pixel_ll = logpdf(Poisson(rate[1]), round(Int, this_pixel))
        ll += pixel_ll
    end
    return ll
end


function joint_log_prob(images::Vector{TiledImage},
                        active_pixels::Vector{ActivePixel},
                        states::Vector{LatentState},
                        ea::ElboArgs,
                        prior::Prior)

    # log prior for state variables
    log_prior = state_log_prior(states, prior)

    # log like
    log_like = state_log_likelihood(states, images, active_pixels, ea)

    println("log prob: ", log_prior, " ", log_like)
    return log_prior + log_like
end


function one_node_infer_mcmc(rcfs::Vector{RunCamcolField},
                             stagedir::String;
                             objid="",
                             box=BoundingBox(-1000., 1000., -1000., 1000.),
                             primary_initialization=true)
    # catalog
    duplicate_policy = primary_initialization ? :primary : :first
    catalog = Celeste.SDSSIO.read_photoobj_files(rcfs, stagedir,
                              duplicate_policy=duplicate_policy)

    # Filter out low-flux objects in the catalog.
    catalog = filter(entry->(maximum(entry.star_fluxes) >= MIN_FLUX), catalog)
    println("$(length(catalog)) primary sources after MIN_FLUX cut")

    # Filter any object not specified, if an objid is specified
    if objid != ""
        Log.info(catalog[1].objid)
        catalog = filter(entry->(entry.objid == objid), catalog)
    end

    # Get indicies of entries in the  RA/Dec range of interest.
    entry_in_range = entry->((box.ramin < entry.pos[1] < box.ramax) &&
                             (box.decmin < entry.pos[2] < box.decmax))
    target_sources = find(entry_in_range, catalog)

    # If there are no objects of interest, return early.
    if length(target_sources) == 0
        return Dict{Int, Dict}()
    end

    # Read in images for all (run, camcol, field).
    images = load_images(rcfs, stagedir)

    println("finding neighbors")
    neighbor_map = Celeste.Infer.find_neighbors(target_sources, catalog, images)

    # iterate over sources
    curr_source = 1
    ts    = curr_source
    s     = target_sources[ts]
    entry     = catalog[s]
    neighbors = [catalog[m] for m in neighbor_map[s]]

    # generate samples for source entry/neighbors pair
    samples = run_single_source_sampler(entry, neighbors, images)
    return samples
end


#
# util
#

function load_prior()
    pp = Model.load_prior()

    star_prior = StarPrior(
        LogNormal(pp.r_mean[1], pp.r_var[1]),
        Categorical(pp.k[:,1]),
        [MvLogNormal(pp.c_mean[:,k,1], pp.c_cov[:,:,k,1]) for k in 1:2])

    gal_prior = GalaxyPrior(
        LogNormal(pp.r_mean[2], pp.r_var[2]),
        Categorical(pp.k[:,2]),
        [MvLogNormal(pp.c_mean[:,k,2], pp.c_cov[:,:,k,2]) for k in 1:2],
        LogNormal(0, 10),
        Beta(1, 1),
        Beta(1, 1))

    return Prior(Bernoulli(.5), star_prior, gal_prior)
end

function logsumexp(a::Vector{Float64})
    a_max = maximum(a)
    out = log(sum(exp(a - a_max)))
    out += a_max
    return out
end

function colors_to_fluxes(brightness::Float64, colors::Vector{Float64})
    ret    = Array(Float64, 5)
    ret[3] = exp(brightness)
    ret[4] = ret[3] * exp(colors[3]) #vs[ids.c1[3, i]])
    ret[5] = ret[4] * exp(colors[4]) #vs[ids.c1[4, i]])
    ret[2] = ret[3] / exp(colors[2]) #vs[ids.c1[2, i]])
    ret[1] = ret[2] / exp(colors[1]) #vs[ids.c1[1, i]])
    ret
end

# Main test entry point
function run_gibbs_sampler_fixed()
    # very small patch of sky that turns out to have 4 sources.
    # We checked that this patch is in the given field.
    box            = Celeste.BoundingBox(200., 200.5, 38.1, 38.35)
    field_triplets = [RunCamcolField(3900, 2, 453),]
    stagedir       = joinpath(ENV["SCRATCH"], "celeste")
    samples        = one_node_infer_mcmc(field_triplets, stagedir; box=box)
end

run_gibbs_sampler_fixed()
