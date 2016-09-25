#!/usr/bin/env julia

module GalsimBenchmark

using DataFrames
import FITSIO
import WCS

import Celeste: Infer, Model, DeterministicVI

IOTA = 1000.

## TODO pull this from test/SampleData.jl
function make_elbo_args(images::Vector{Model.TiledImage},
                        catalog::Vector{Model.CatalogEntry};
                        fit_psf::Bool=false)
    vp = Vector{Float64}[Model.init_source(ce) for ce in catalog]
    patches, tile_source_map = Infer.get_tile_source_map(images, catalog)
    active_sources = collect(1:length(catalog))
    ea = DeterministicVI.ElboArgs(images, vp, tile_source_map, patches, active_sources)
    if fit_psf
        Infer.fit_object_psfs!(ea, ea.active_sources)
    end
    ea
end

## end code from SampleData

function make_psf()
    alphaBar = [1.; 1.; 1.] ./ 3
    xiBar = [0.; 0.]
    tauBar = [1. 0.; 0. 1.]
    [Model.PsfComponent(alphaBar[k], xiBar, tauBar) for k in 1:3]
end

function read_fits(filename; read_sdss_psf=false)
    @assert isfile(filename)
    fits = FITSIO.FITS(filename)
    pixels = read(fits[1])
    header_str = FITSIO.read_header(fits[1], String)
    wcs = WCS.from_header(header_str)[1]
    if read_sdss_psf
        header = FITSIO.read_header(fits[1])
        psf = read_psf_from_sdss_header(header)
    else
        psf = make_psf()
    end
    close(fits)
    pixels, psf, wcs
end

function make_band_images(band_pixels, band_psfs, wcs, epsilon)
    H, W = size(band_pixels[1])
    [
        Model.TiledImage(
            Model.Image(
                H,
                W,
                band_pixels[band],
                band,
                wcs,
                band_psfs[band],
                0, # SDSS run
                0, # SDSS camcol
                0, # SDSS field
                fill(epsilon, H, W), #epsilon_mat,
                fill(IOTA, H), #iota_vec,
                Model.RawPSF(Array(Float64, 0, 0), 0, 0, Array(Float64, 0, 0, 0)),
            ),
            tile_width=48,
        )
        for band in 1:5
    ]
end

function make_catalog_entry()
    Model.CatalogEntry(
        [18., 18.], # pos
        false, # is_star
        fill(100., 5), #sample_star_fluxes
        fill(1000., 5), #sample_galaxy_fluxes
        0.1, # gal_frac_dev
        0.7, # gal_ab
        pi / 4, # gal_angle
        4., # gal_scale
        "sample", # objid
        0, # thing_id
    )
end

function canonical_angle(params)
    angle_radians = params[Model.ids.e_angle]
    while angle_radians < 0
        angle_radians += 2 * pi
    end
    while angle_radians > 2 * pi
        angle_radians -= 2 * pi
    end
    angle_radians
end

function pretty_print_params(params, truth_row)
    ids = Model.ids
    benchmark_data = DataFrame(
        field=String[
            "X center (world coords)",
            "Y center (world coords)",
            "Weight on exponential (vs. Vaucouleurs)",
            "Minor/major axis ratio",
            "Angle (degrees)",
            "Half-light radius (arcsec) (TODO)",
            "Galaxy brightness (nMgy) (TODO)",
            "Probability of galaxy (TODO)",
        ],
        expected=Float64[
            truth_row[1, :world_center_x],
            truth_row[1, :world_center_y],
            1,
            truth_row[1, :minor_major_axis_ratio],
            truth_row[1, :angle_degrees],
            truth_row[1, :half_light_radius_arcsec],
            truth_row[1, :flux_counts] / IOTA,
            1,
        ],
        actual=Float64[
            params[ids.u[1]],
            params[ids.u[2]],
            params[ids.e_dev],
            params[ids.e_axis],
            canonical_angle(params) * 180 / pi,
            params[ids.e_scale],
            exp(params[ids.r1[2]]),
            params[ids.a[2]],
        ],
    )
    println(truth_row[1, :comment])
    println(repr(benchmark_data))
end

function main(; verbose=false)
    truth_data = readtable("galsim_truth.csv")
    for index in 0:(size(truth_data, 1) - 1)
        filename = "output/galsim_test_image_$index.fits"
        println("Reading $filename...")
        pixels, psf, wcs = read_fits(filename)
        epsilon = truth_data[index + 1, :sky_level] / IOTA
        band_images::Vector{Model.TiledImage} =
            make_band_images(fill(pixels, 5), fill(psf, 5), wcs, epsilon)
        catalog_entry::Model.CatalogEntry = make_catalog_entry()

        elbo_args::DeterministicVI.ElboArgs =
            make_elbo_args(band_images, [catalog_entry], fit_psf=false)
        DeterministicVI.maximize_f(DeterministicVI.elbo, elbo_args, verbose=verbose)
        variational_parameters::Vector{Float64} = elbo_args.vp[1]
        pretty_print_params(variational_parameters, truth_data[index + 1, :])
    end
end

end # module GalsimBenchmark
