#!/usr/bin/env julia

# ./celeste.jl infer-box 200 200.5 38.1 38.35
using Distributions
import Celeste: Model, Infer, RunCamcolField, load_images, init_source
import Celeste.Model: CatalogEntry, TiledImage, ElboArgs,
                      get_active_pixels

# Mamba for MCMC tools
import Mamba

const MIN_FLUX = 2.0

type SampleResult
    star_samples::Array{Float64, 2}
    galaxy_samples::Array{Float64, 2}
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
    ea = ElboArgs(images, vp, tile_source_map, patches, [1])
    Infer.fit_object_psfs!(ea, ea.active_sources)
    Infer.trim_source_tiles!(ea)
    active_pixels = get_active_pixels(ea)

    # generate the star logpdf
    # TODO create a sample_from_star_prior function (over dispersion...)
    star_logpdf, star_logprior = Model.make_star_logpdf(images, active_pixels, ea)
    star_state  = [.1 for i in 1:7]
    println("Star logpdf: ", star_logpdf(star_state))
    #star_state   = init_star_state()
    #star_samples, star_lls = run_slice_sampler(star_logpdf, star_state)
    star_param_names = ["lnr", "cug", "cgr", "cri", "ciz", "ra", "dec"]
    sim = run_slice_sampler(star_logpdf, star_state, 400, star_param_names)

    # generate the galaxy logpdf
    # TODO create a sample_from_gal_prior function (with over dispersion)
    #gal_state   = init_gal_state()
    gal_logpdf, gal_logprior = Model.make_galaxy_logpdf(images, active_pixels, ea)
    gal_state = [.1 for i in 1:11]
    println("Gal logpdf: ", gal_logpdf(gal_state))
    #gal_param_names = [star_param_names[1:7]; ["gdev", "gaxis", "gangle", "gscale"]]
    #gal_samples, gal_lls  = run_slice_sampler(gal_logpdf, gal_state)

    # generate pointers to star/gal type (to infer p(star | data))
    #type_samples, type_lls = run_star_gal_switcher(star_lls, gal_lls)

    # return all samples
    #return SampleResult(star_samples, gal_samples, type_samples,
    #                    star_lls, gal_lls, type_lls)
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
    sim = Mamba.Chains(N, 7, names = param_names)
    th  = Mamba.SliceMultivariate(th0, 5., lnpdf)
    for i in 1:N
      sample!(th)
      sim[i, :, 1] = th
    end
    Mamba.describe(sim)
    return sim
end

function run_star_gal_switcher(star_lls::Vector{Float64},
                               gal_lls::Vector{Float64})
    #TODO switch between states of the chain, alt to RJMCMC
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
