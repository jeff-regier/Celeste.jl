#!/usr/bin/env julia

module GalsimBenchmark

using DataFrames
import FITSIO
import StaticArrays
import WCS

import Celeste: Infer, Model, DeterministicVI

const IOTA = 1000.
const FILENAME = "output/galsim_test_images.fits"

function make_psf()
    alphaBar = [1.; 1.; 1.] ./ 3
    xiBar = [0.; 0.]
    tauBar = [1. 0.; 0. 1.]
    [
        Model.PsfComponent(
            alphaBar[k],
            StaticArrays.SVector{2, Float64}(xiBar),
            StaticArrays.SMatrix{2, 2, Float64, 4}(tauBar)
        )
        for k in 1:3
    ]
end

function read_fits(filename; read_sdss_psf=false)
    @assert isfile(filename)
    println("Reading $filename...")
    fits = FITSIO.FITS(filename)
    println("Found $(length(fits)) extensions.")

    multi_extension_pixels = []
    for extension in fits
        push!(multi_extension_pixels, read(extension))
    end

    # assume WCS same for each extension
    header_str = FITSIO.read_header(fits[1], String)
    wcs = WCS.from_header(header_str)[1]

    close(fits)
    multi_extension_pixels, wcs
end

function make_tiled_images(band_pixels, psf, wcs, epsilon)
    # assume dimensions equal for all images
    height, width = size(band_pixels[1])
    [
        Model.TiledImage(
            Model.Image(
                height,
                width,
                band_pixels[band],
                band,
                wcs,
                psf,
                0, # SDSS run
                0, # SDSS camcol
                0, # SDSS field
                fill(epsilon, height, width), #epsilon_mat,
                fill(IOTA, height), #iota_vec,
                Model.RawPSF(Array(Float64, 0, 0), 0, 0, Array(Float64, 0, 0, 0)),
            ),
            tile_width=48,
        )
        for band in 1:5
    ]
end

function typical_band_relative_intensities(is_star::Bool)
    source_type_index = is_star ? 1 : 2
    prior_parameters::Model.PriorParams = Model.load_prior()
    # Band relative intensities are a mixture of lognormals. Which mixture component has the most
    # weight?
    dominant_component = indmax(prior_parameters.k[:, source_type_index])
    # What are the most typical log relative intensities for that component?
    inter_band_ratios = exp(
        prior_parameters.c_mean[:, dominant_component, source_type_index]
        - diag(prior_parameters.c_cov[:, :, dominant_component, source_type_index])
    )
    Float64[
        1 / inter_band_ratios[2] / inter_band_ratios[1],
        1 / inter_band_ratios[2],
        1,
        inter_band_ratios[3],
        inter_band_ratios[3] * inter_band_ratios[4],
    ]
end

function typical_reference_brightness(is_star::Bool)
    source_type_index = is_star ? 1 : 2
    prior_parameters::Model.PriorParams = Model.load_prior()
    exp(
        prior_parameters.r_mean[source_type_index]
        - prior_parameters.r_var[source_type_index]
    )
end

function make_catalog_entry()
    Model.CatalogEntry(
        [18., 18.], # pos
        false, # is_star
        # sample_star_fluxes
        typical_band_relative_intensities(true) .* typical_reference_brightness(true),
        # sample_galaxy_fluxes
        typical_band_relative_intensities(false) .* typical_reference_brightness(false),
        0.1, # gal_frac_dev
        0.7, # gal_ab
        pi / 4, # gal_angle
        4., # gal_scale
        "sample", # objid
        0, # thing_id
    )
end

# Since we're considering elliptical galaxy shapes, angle is only meaningful up to rotations of 180
# deg. This finds an equivalent angle in [0, 180) deg.
function canonical_angle(params)
    angle_radians = params[Model.ids.e_angle]
    while angle_radians < 0
        angle_radians += pi
    end
    while angle_radians > pi
        angle_radians -= pi
    end
    angle_radians
end

const BENCHMARK_PARAMETER_LABELS = String[
    "X center (world coords)",
    "Y center (world coords)",
    "Minor/major axis ratio",
    "Angle (degrees)",
    "Half-light radius (arcsec)",
    "Brightness (nMgy)",
    "Color band 1-2 ratio",
    "Color band 2-3 ratio",
    "Color band 3-4 ratio",
    "Color band 4-5 ratio",
    "Probability of galaxy",
]

function benchmark_comparison_data(params, truth_row)
    ids = Model.ids
    star_galaxy_index = truth_row[1, :star_or_galaxy] == "star" ? 1 : 2
    DataFrame(
        label=fill(truth_row[1, :comment], length(BENCHMARK_PARAMETER_LABELS)),
        field=BENCHMARK_PARAMETER_LABELS,
        expected=Any[
            truth_row[1, :world_center_x],
            truth_row[1, :world_center_y],
            truth_row[1, :minor_major_axis_ratio],
            truth_row[1, :angle_degrees],
            truth_row[1, :half_light_radius_arcsec],
            truth_row[1, :reference_band_flux_nmgy],
            NA,
            NA,
            NA,
            NA,
            truth_row[1, :star_or_galaxy] == "star" ? 0 : 1,
        ],
        actual=Float64[
            params[ids.u[1]],
            params[ids.u[2]],
            params[ids.e_axis],
            canonical_angle(params) * 180 / pi,
            params[ids.e_scale],
            exp(params[ids.r1[star_galaxy_index]]),
            exp(params[ids.c1[1, star_galaxy_index]]),
            exp(params[ids.c1[2, star_galaxy_index]]),
            exp(params[ids.c1[3, star_galaxy_index]]),
            exp(params[ids.c1[4, star_galaxy_index]]),
            params[ids.a[2]],
        ],
    )
end

function main(; verbose=false)
    truth_data = readtable("galsim_truth.csv")
    all_benchmark_data = []
    psf = make_psf()
    multi_extension_pixels::Vector{Matrix{Float32}}, wcs = read_fits(FILENAME)
    for index in 1:size(truth_data, 1)
        epsilon = truth_data[index, :sky_level_nmgy]

        first_band_index = (index - 1) * 5 + 1
        band_pixels = multi_extension_pixels[first_band_index:(first_band_index + 4)]
        band_images::Vector{Model.TiledImage} = make_tiled_images(band_pixels, psf, wcs, epsilon)
        catalog_entry::Model.CatalogEntry = make_catalog_entry()

        if truth_data[index, :add_noise] == 0
            expected_flux = (
                (truth_data[index, :reference_band_flux_nmgy]
                 + prod(size(band_pixels[3])) * truth_data[index, :sky_level_nmgy]) * IOTA
            )
            @assert abs(sum(band_pixels[3]) - expected_flux) / expected_flux < 1e-3
        end

        vp = Vector{Float64}[Model.init_source(catalog_entry)]
        patches, tile_source_map = Infer.get_tile_source_map(band_images, [catalog_entry])
        elbo_args = DeterministicVI.ElboArgs(band_images, vp, tile_source_map, patches, [1])
        Infer.load_active_pixels!(elbo_args)
        DeterministicVI.maximize_f(DeterministicVI.elbo, elbo_args, verbose=verbose, loc_width=3.0)
        variational_parameters::Vector{Float64} = vp[1]

        benchmark_data = benchmark_comparison_data(variational_parameters, truth_data[index, :])
        println(repr(benchmark_data))
        push!(all_benchmark_data, benchmark_data)
    end

    println(repr(vcat(all_benchmark_data...)))
end

end # module GalsimBenchmark
