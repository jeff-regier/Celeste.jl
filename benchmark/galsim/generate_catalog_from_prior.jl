#!/usr/bin/env julia

using Distributions
import FITSIO

import Celeste.Model

# This is the ratio of stars derived from the catalog used the generate the prior; the value 0.99 is
# currently used in inference to account for the extra flexibility of the galaxy model
const PRIOR_PROBABILITY_OF_STAR = 0.28
const FITS_CATALOG_FILENAME = joinpath("output", "galsim_field_500_catalog.fits")
# Celeste's e_scale is in pixels (hence the prior is in pixels), but this is tied to the SDSS
# resolution. It makes more sense to save a catalog with radii in arcsec, so we do that conversion.
const SDSS_ARCSEC_PER_PIXEL = 0.396

immutable SourceParams
    relative_position_x::Float64
    relative_position_y::Float64
    is_star::Bool
    reference_band_flux_nmgy::Float64
    color_ratio_21::Float64
    color_ratio_32::Float64
    color_ratio_43::Float64
    color_ratio_54::Float64
    half_light_radius_arcsec::Float64
    angle_deg::Float64
    minor_major_axis_ratio::Float64
    de_vaucouleurs_mixture_weight::Float64
end

function draw_source_params(prior)
    is_star = (rand(Bernoulli(PRIOR_PROBABILITY_OF_STAR)) == 1)
    source_type_index = is_star ? 1 : 2
    reference_band_flux_nmgy = exp(rand(
        Normal(prior.r_μ[source_type_index], sqrt(prior.r_σ²[source_type_index]))
    ))

    color_mixture_weights = prior.k[:, source_type_index]
    num_color_components = length(color_mixture_weights)
    color_components = MvNormal[
        MvNormal(prior.c_mean[:, k, source_type_index], prior.c_cov[:, :, k, source_type_index])
        for k in 1:num_color_components
    ]
    color_ratios = exp(rand(MixtureModel(color_components, color_mixture_weights)))

    if !is_star
        half_light_radius_px = exp(rand(
            Normal(prior.r_μ[source_type_index], sqrt(prior.r_σ²[source_type_index]))
        ))
        angle_deg = rand(Uniform(0, 180))
        minor_major_axis_ratio = rand(Beta(2, 2))
        de_vaucouleurs_mixture_weight = rand(Beta(0.5, 0.5))
    else
        half_light_radius_px = -1
        angle_deg = -1
        minor_major_axis_ratio = -1
        de_vaucouleurs_mixture_weight = -1
    end

    position_x = rand(Uniform(0, 1))
    position_y = rand(Uniform(0, 1))

    SourceParams(
        position_x,
        position_y,
        is_star,
        reference_band_flux_nmgy,
        color_ratios[1],
        color_ratios[2],
        color_ratios[3],
        color_ratios[4],
        half_light_radius_px * SDSS_ARCSEC_PER_PIXEL,
        angle_deg,
        minor_major_axis_ratio,
        de_vaucouleurs_mixture_weight,
    )
end

function write_fits_table(filename::String, all_sources::Vector{SourceParams})
    println("Writing catalog to $filename")
    fits_stream = FITSIO.FITS(filename, "w")
    columns = Dict(
        string(field_name) => [getfield(source, field_name) for source in all_sources]
        for field_name in fieldnames(SourceParams)
    )
    if !isdir(dirname(filename))
        println("Creating directory $(dirname(filename))")
        mkdir(dirname(filename))
    end
    write(fits_stream, columns)
    close(fits_stream)
end

function main()
    srand(12345)
    prior = Model.load_prior()
    source_params = [draw_source_params(prior) for index in 1:500]
    write_fits_table(FITS_CATALOG_FILENAME, source_params)
end

main()
