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
    alphaBar = [1.; 0.]
    xiBar = [0.; 0.]
    tauBar = [1. 0.; 0. 1.]
    [
        Model.PsfComponent(
            alphaBar[k],
            StaticArrays.SVector{2, Float64}(xiBar),
            StaticArrays.SMatrix{2, 2, Float64, 4}(tauBar)
        )
        for k in 1:2
    ]
end

immutable FitsExtension
    pixels::Matrix{Float32}
    header::FITSIO.FITSHeader
end

function read_fits(filename; read_sdss_psf=false)
    @assert isfile(filename)
    println("Reading $filename...")
    fits = FITSIO.FITS(filename)
    println("Found $(length(fits)) extensions.")

    extensions::Vector{FitsExtension} = []
    for extension in fits
        push!(extensions, FitsExtension(read(extension), FITSIO.read_header(extension)))
    end

    # assume WCS same for each extension
    wcs = WCS.from_header(FITSIO.read_header(fits[1], String))[1]

    close(fits)
    extensions, wcs
end

function make_tiled_images(band_pixels, psf, wcs, epsilon, iota)
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
                fill(epsilon, height, width),
                fill(iota, height),
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

function get_field(header::FITSIO.FITSHeader, key::String)
    if haskey(header, key)
        header[key]
    else
        NA
    end
end

function benchmark_comparison_data(params, header)
    ids = Model.ids
    star_galaxy_index = header["CL_STGAL"] == "star" ? 1 : 2
    DataFrame(
        label=fill(header["CL_DESCR"], length(BENCHMARK_PARAMETER_LABELS)),
        field=BENCHMARK_PARAMETER_LABELS,
        expected=Any[
            get_field(header, "CL_CENTX"),
            get_field(header, "CL_CENTY"),
            get_field(header, "CL_RATIO"),
            get_field(header, "CL_ANGLE"),
            get_field(header, "CL_HLRAD"),
            get_field(header, "CL_FLUX"),
            get_field(header, "CL_COL12"),
            get_field(header, "CL_COL23"),
            get_field(header, "CL_COL34"),
            get_field(header, "CL_COL45"),
            header["CL_STGAL"] == "star" ? 0 : 1,
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

function assert_counts_match_expected_flux(band_pixels::Vector{Matrix{Float32}},
                                           header::FITSIO.FITSHeader)
    if !header["CL_NOISE"]
        expected_flux = (
            (header["CL_FLUX"] + prod(size(band_pixels[3])) * header["CL_SKY"]) * iota
        )
        @assert abs(sum(band_pixels[3]) - expected_flux) / expected_flux < 1e-3
    end
end

# this code is very close to Infer.infer_source() but avoids PSF fitting
function infer_source(band_images::Vector{Model.TiledImage},
                      catalog_entry::Model.CatalogEntry,
                      verbose::Bool)
    vp = Vector{Float64}[Model.init_source(catalog_entry)]
    patches, tile_source_map = Infer.get_tile_source_map(band_images, [catalog_entry])
    elbo_args = DeterministicVI.ElboArgs(band_images, vp, tile_source_map, patches, [1])
    Infer.load_active_pixels!(elbo_args)
    DeterministicVI.maximize_f(DeterministicVI.elbo, elbo_args, verbose=verbose, loc_width=3.0)
    variational_parameters::Vector{Float64} = vp[1]
    variational_parameters
end

function main(; verbose=false)
    all_benchmark_data = []
    psf = make_psf()
    extensions, wcs = read_fits(FILENAME)
    @assert length(extensions) % 5 == 0 # one extension per band for each test case

    for test_case_index in 1:div(length(extensions), 5)
        first_band_index = (test_case_index - 1) * 5 + 1
        header = extensions[first_band_index].header

        band_pixels = [
            extensions[index].pixels for index in first_band_index:(first_band_index + 4)
        ]
        assert_counts_match_expected_flux(band_pixels, header)
        band_images::Vector{Model.TiledImage} =
            make_tiled_images(band_pixels, psf, wcs, header["CL_SKY"], header["CL_IOTA"])
        catalog_entry::Model.CatalogEntry = make_catalog_entry()

        variational_parameters = infer_source(band_images, catalog_entry, verbose)

        benchmark_data = benchmark_comparison_data(variational_parameters, header)
        println(repr(benchmark_data))
        push!(all_benchmark_data, benchmark_data)
    end

    full_data = vcat(all_benchmark_data...)
    println(repr(full_data[!isna(full_data[:expected]), :]))
end

end # module GalsimBenchmark
