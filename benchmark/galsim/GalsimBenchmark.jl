#!/usr/bin/env julia

module GalsimBenchmark

using DataFrames
import FITSIO
import WCS

import Celeste: ElboDeriv, Infer, Model, OptimizeElbo

## TODO pull this from test/SampleData.jl
function make_elbo_args(images::Vector{Model.TiledImage},
                        catalog::Vector{Model.CatalogEntry};
                        fit_psf::Bool=false)
    vp = Vector{Float64}[Model.init_source(ce) for ce in catalog]
    patches, tile_source_map = Infer.get_tile_source_map(images, catalog)
    active_sources = collect(1:length(catalog))
    ea = ElboDeriv.ElboArgs(images, vp, tile_source_map, patches, active_sources)
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

function make_band_images(band_pixels, band_psfs, wcs)
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
                fill(0., H, W), #epsilon_mat,
                fill(1000., H), #iota_vec,
                Model.RawPSF(Array(Float64, 0, 0), 0, 0, Array(Float64, 0, 0, 0)),
            ),
            tile_width=48,
        )
        for band in 1:5
    ]
end

# TODO: from test/SampleData.jl
const sample_star_fluxes = fill(1000., 5)
const sample_galaxy_fluxes = fill(100., 5)

function make_catalog_entry()
    Model.CatalogEntry(
        [18., 18.], # pos
        false, # is_star
        sample_star_fluxes,
        sample_galaxy_fluxes,
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
    @printf "Location in world coords: (%.2f, %.2f)\n" params[ids.u[1]] params[ids.u[2]]
    @printf "  Expected (%.2f, %.2f)\n" truth_row[1, :world_center_x] truth_row[1, :world_center_y]
    @printf "Weight on exponential (vs. Vaucouleurs): %.2f\n" params[ids.e_dev]
    @printf "Minor/major axis ratio: %.2f\n" params[ids.e_axis]
    angle_radians = canonical_angle(params)
    @printf "Angle: %.2f rad (%.1f deg)\n" angle_radians angle_radians * 180 / pi
    @printf "Scale: %.2f\n" params[ids.e_scale]
    @printf "Galaxy brightness lognormal mean %.2f, var %.2f\n" params[ids.r1[2]] params[ids.r2[2]]
    @printf "Probability of star: %.2f; of galaxy: %.2f\n" params[ids.a[1]] params[ids.a[2]]
end

function main(; verbose=false)
    truth_df = readtable("galsim_truth.csv")
    for index in 0:size(truth_df, 1)
        filename = "output/galsim_test_image_$index.fits"
        println("Reading $filename...")
        pixels, psf, wcs = read_fits(filename)
        band_images = make_band_images(fill(pixels, 5), fill(psf, 5), wcs)
        catalog_entry = make_catalog_entry()

        ea = make_elbo_args(band_images, [catalog_entry], fit_psf=false)
        iter_count, max_f, max_x, nm_result = OptimizeElbo.maximize_f(
            ElboDeriv.elbo_likelihood,
            ea,
            loc_width=3.0,
            verbose=verbose,
        )
        pretty_print_params(ea.vp[1], truth_df[index + 1, :])
    end
end

end # module GalsimBenchmark
