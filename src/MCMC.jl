module MCMC

using ..Model
import Mamba

# TODO move these to model/log_prob.jl
star_param_names = ["lnr", "cug", "cgr", "cri", "ciz", "ra", "dec"]
galaxy_param_names = [star_param_names; ["gdev", "gaxis", "gangle", "gscale"]]


function run_single_star_mcmc(num_samples::Int64,
                              sources::Vector{CatalogEntry},
                              images::Vector{TiledImage},
                              active_pixels::Vector{ActivePixel},
                              S::Int64,
                              N::Int64,
                              tile_source_map::Vector{Matrix{Vector{Int}}},
                              patches::Matrix{SkyPatch},
                              active_sources::Vector{Int},
                              psf_K::Int64,
                              num_allowed_sd::Float64)

    # turn list of catalog entries a list of LatentStateParams
    # and create logpdf function handle
    source_states = [Model.catalog_entry_to_latent_state_params(s)
                     for s in sources]
    star_logpdf, star_logprior =
        Model.make_star_logpdf(images, active_pixels, S, N,
                               source_states, tile_source_map,
                               patches, active_sources,
                               psf_K, num_allowed_sd)

    # initialize star params
    star_state = Model.extract_star_state(source_states[1])
    println(star_state)
    best_ll = star_logpdf(star_state)

    # run star slic esampler for 
    star_sim = run_slice_sampler(star_logpdf, star_state,
                                 num_samples, star_param_names)
    star_sim
end


function run_single_galaxy_mcmc(num_samples::Int64,
                                sources::Vector{CatalogEntry},
                                images::Vector{TiledImage},
                                active_pixels::Vector{ActivePixel},
                                S::Int64,
                                N::Int64,
                                tile_source_map::Vector{Matrix{Vector{Int}}},
                                patches::Matrix{SkyPatch},
                                active_sources::Vector{Int},
                                psf_K::Int64,
                                num_allowed_sd::Float64)

    source_states = [Model.catalog_entry_to_latent_state_params(s)
                     for s in sources]
    gal_logpdf, gal_logprior =
        Model.make_galaxy_logpdf(images, active_pixels, S, N,
                                 source_states, tile_source_map,
                                 patches, active_sources,
                                 psf_K, num_allowed_sd)

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


end
