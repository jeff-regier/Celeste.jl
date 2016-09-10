#!/usr/bin/env julia

module GalsimBenchmark

import FITSIO
import WCS

import Celeste: ElboDeriv, Infer, Model, OptimizeElbo

# TODO pull this from test/SampleData.jl
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



const FILENAME = "output/galsim_test_image.fits"

function read_fits()
    fits = FITSIO.FITS(FILENAME)
    hdr = FITSIO.read_header(fits[1])
    pixels = read(fits[1])
    header_str = FITSIO.read_header(fits[1], String)
    wcs = WCS.from_header(header_str)[1]
    close(fits)
    pixels, wcs
end

function make_psf()
    alphaBar = [1.; 0.; 0.]
    xiBar = [0.; 0.]
    tauBar = [1. 0.; 0. 1.]
    [Model.PsfComponent(alphaBar[k], xiBar, tauBar) for k in 1:3]
end

function make_band_images(pixels, wcs)
    H, W = size(pixels)
    [
        Model.TiledImage(
            Model.Image(
                H,
                W,
                pixels,
                band,
                wcs,
                make_psf(),
                0, # SDSS run
                0, # SDSS camcol
                0, # SDSS field
                fill(0., H, W), #epsilon_mat,
                fill(950., H), #iota_vec,
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
    pixels, wcs = read_fits()
    band_images = make_band_images(pixels, wcs)
    ea = make_elbo_args(band_images, [make_catalog_entry()], fit_psf=false)
    iter_count, max_f, max_x, nm_result = OptimizeElbo.maximize_f(
        ElboDeriv.elbo_likelihood,
        ea,
        loc_width=1.0,
    )
    pretty_print_params(max_x)
    max_x
end

end # module GalsimBenchmark
