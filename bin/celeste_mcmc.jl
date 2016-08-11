#!/usr/bin/env julia

using Distributions

using Celeste
import Celeste.Model: PsfComponent, psf_K, galaxy_prototypes, D, Ia, prior
using Celeste: Model, ElboDeriv, Infer
import Celeste: WCSUtils, PSF
import Celeste.ElboDeriv: ActivePixel


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


function joint_log_prob(images::Vector{TiledImage},
                        active_pixels::Vector{ActivePixel},
                        states::Vector{LatentState},
                        ea::ElboArgs,
                        params::ModelParams, prior)
    # prior

    log_prior = 0.0
    for s in active_sources
        state = states[s]
        log_prior += logpdf(prior.is_star, state.is_star ? 1 : 0)

        subprior = state.is_star ? star_prior : galaxy_prior
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

    # likelihood

    ll = 0.0

    #TODO: load `star_mcs` and `gal_mcs`, as shown in `ElboDeriv.process_active_pixels!`

    # iterate over the pixels
    for pixel in active_pixels
        tile = ea.images[pixel.n].tiles[pixel.tile_ind]
        tile_sources = ea.tile_source_map[pixel.n][pixel.tile_ind]
        this_pixel = tile.pixels[pixel.h, pixel.w]

        # TODO: set `calculate_deriviatives` and `calculate_hessian` to false in `elbo_vars`
        # TODO: load wcs_jacobian into `ea`
        populate_fsm_vecs!(elbo_vars, ea, tile_sources, tile, h, w, 
                                              sbs, gal_mcs, star_mcs)

        rate = tile.epsilon_mat[pixel.h, pixel.w]
        for s in tile_sources
            state = states[s]
            rate += state.is_star ? elbo_vars.fs0m_vec[s].v : elbo_vars.fs1m_vec[s].v
        end
        rate *= tile.iota_vec[pixel.h]

        ll += logpdf(Poisson(rate), this_pixel)
    end


    # joint log prob
    log_prior + ll
end


function run_gibbs_sampler()
    # TODO: load priors from `Model.prior`

    # preprocssing
    cat_local = vcat(params.entry, params.neighbors)
    vp = Vector{Float64}[init_source(ce) for ce in cat_local]
    patches, tile_source_map = get_tile_source_map(images, cat_local)
    ea = ElboArgs(images, vp, tile_source_map, patches, [1])
    fit_object_psfs!(ea, ea.active_sources)
    trim_source_tiles!(ea)
    active_pixels = ElboDeriv.get_active_pixels(ea)

    for iter in 1:10
        lp = joint_log_prob(images, active_pixels, states, params, prior)
        println(lp)
        # TODO: mutate state based on lp, a markov chain
    end
end

