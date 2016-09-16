#!/usr/bin/env julia

module GalsimBenchmark

import DataFrames # TODO remove after temporary code goes
import FITSIO
import WCS

import Celeste: ElboDeriv, Infer, Model, OptimizeElbo

## TODO pull this from test/SampleData.jl
function make_elbo_args(images::Vector{Model.TiledImage},
                        catalog::Vector{Model.CatalogEntry};
                        fit_psf::Bool=false,
                        patch_radius::Float64=NaN)
    vp = Vector{Float64}[Model.init_source(ce) for ce in catalog]
    patches, tile_source_map = Infer.get_tile_source_map(
        images,
        catalog,
        radius_override=patch_radius,
    )
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
const sample_star_fluxes = [
    4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]
const sample_galaxy_fluxes = [
    1.377666E+01, 5.635334E+01, 1.258656E+02,
    1.884264E+02, 2.351820E+02] * 100  # 1x wasn't bright enough

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

function pretty_print_params(params)
    ids = Model.ids
    @printf "Location in world coords: (%.2f, %.2f)\n" params[ids.u[1]] params[ids.u[2]]
    @printf "Weight on exponential (vs. Vaucouleurs): %.2f\n" params[ids.e_dev]
    @printf "Minor/major axis ratio: %.2f\n" params[ids.e_axis]
    @printf "Angle: %.2f rad (%.1f deg)\n" params[ids.e_angle] params[ids.e_angle] * 180/pi
    @printf "Scale: %.2f\n" params[ids.e_scale]
    @printf "Galaxy brightness lognormal mean %.2f, var %.2f\n" params[ids.r1[2]] params[ids.r2[2]]
    @printf "Probability of star: %.2f; of galaxy: %.2f\n" params[ids.a[1]] params[ids.a[2]]
end

function main()
    pixels, psf, wcs = read_fits("output/galsim_test_image.fits")
    band_images = make_band_images(fill(pixels, 5), fill(psf, 5), wcs)
    catalog_entry = make_catalog_entry()

    ea = make_elbo_args(band_images, [catalog_entry], fit_psf=false)
    iter_count, max_f, max_x, nm_result = OptimizeElbo.maximize_f(
        ElboDeriv.elbo_likelihood,
        ea,
        loc_width=1.0,
    )
    pretty_print_params(ea.vp[1])
    ea.vp
end

end # module GalsimBenchmark
