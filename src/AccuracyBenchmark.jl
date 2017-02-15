module AccuracyBenchmark

using DataFrames
using Distributions
import FITSIO
import StaticArrays
import WCS

import Celeste.Model
import Celeste.ParallelRun
import Celeste.SDSSIO

const SDSS_ARCSEC_PER_PIXEL = 0.396
const ARCSEC_PER_DEGREE = 3600

type BenchmarkFitsFileNotFound <: Exception
    filename::String
end

"""
convert between SDSS mags and SDSS flux (nMgy)
"""
mag_to_flux(m::AbstractFloat) = 10.^(0.4 * (22.5 - m))
flux_to_mag(nm::AbstractFloat) = nm > 0 ? 22.5 - 2.5 * log10(nm) : NaN

function color_from_fluxes(flux1::AbstractFloat, flux2::AbstractFloat)
    if flux1 <= 0 || flux2 <= 0
        NA
    else
        log(flux2 / flux1)
    end
end
function color_from_mags(mags1::AbstractFloat, mags2::AbstractFloat)
    color_from_fluxes(mag_to_flux(mags1), mag_to_flux(mags1))
end

STRIPE_82_CATALOG_KEYS = [
    :objid, :rerun, :run, :camcol, :field, :flags,
    :ra, :dec, :probpsf,
    :psfmag_u, :psfmag_g, :psfmag_r, :psfmag_i, :psfmag_z,
    :devmag_u, :devmag_g, :devmag_r, :devmag_i, :devmag_z,
    :expmag_u, :expmag_g, :expmag_r, :expmag_i, :expmag_z,
    :fracdev_r,
    :devab_r, :expab_r,
    :devphi_r, :expphi_r,
    :devrad_r, :exprad_r,
    :flags,
]

function load_stripe82_fits_catalog_as_data_frame(filename, extension_index)
    result_df = DataFrame()
    fits = FITSIO.FITS(filename)
    try
        for key in STRIPE_82_CATALOG_KEYS
            result_df[key] = read(fits[extension_index], string(key))
        end
        return result_df
    finally
        close(fits)
    end
end

"""
Load Stripe 82 objects into a DataFrame. `fits_filename` should be a FITS file
created by running a CasJobs (skyserver.sdss.org/casjobs/) query
on the Stripe82 database. Run the following query in the \"Stripe82\"
context, then download the table as a FITS file.

```
select
  objid, rerun, run, camcol, field, flags,
  ra, dec, probpsf,
  psfmag_u, psfmag_g, psfmag_r, psfmag_i, psfmag_z,
  devmag_u, devmag_g, devmag_r, devmag_i, devmag_z,
  expmag_u, expmag_g, expmag_r, expmag_i, expmag_z,
  fracdev_r,
  devab_r, expab_r,
  devphi_r, expphi_r,
  devrad_r, exprad_r
into mydb.s82_0_1_0_1
from stripe82.photoobj
where
  run in (106, 206) and
  ra between 0. and 1. and
  dec between 0. and 1.
```
"""
function load_coadd_catalog(fits_filename)
    raw_df = load_stripe82_fits_catalog_as_data_frame(fits_filename, 2)

    usedev = raw_df[:fracdev_r] .> 0.5  # true=> use dev, false=> use exp
    dev_or_exp(dev_column, exp_column) = ifelse(usedev, raw_df[dev_column], raw_df[exp_column])
    is_star = [x != 0 for x in raw_df[:probpsf]]
    function star_or_galaxy(star_column, galaxy_dev_column, galaxy_exp_column)
        ifelse(is_star, raw_df[star_column], dev_or_exp(galaxy_dev_column, galaxy_exp_column))
    end

    mag_u = star_or_galaxy(:psfmag_u, :devmag_u, :expmag_u)
    mag_g = star_or_galaxy(:psfmag_g, :devmag_g, :expmag_g)
    mag_r = star_or_galaxy(:psfmag_r, :devmag_r, :expmag_r)
    mag_i = star_or_galaxy(:psfmag_i, :devmag_i, :expmag_i)
    mag_z = star_or_galaxy(:psfmag_z, :devmag_z, :expmag_z)

    result = DataFrame()
    result[:objid] = raw_df[:objid]
    result[:right_ascension_deg] = raw_df[:ra]
    result[:declination_deg] = raw_df[:dec]
    result[:is_star] = is_star

    result[:reference_band_flux_nmgy] = mag_to_flux.(mag_r)

    result[:color_log_ratio_ug] = color_from_mags.(mag_u, mag_g)
    result[:color_log_ratio_gr] = color_from_mags.(mag_g, mag_r)
    result[:color_log_ratio_ri] = color_from_mags.(mag_r, mag_i)
    result[:color_log_ratio_iz] = color_from_mags.(mag_i, mag_z)

    # gal shape -- fracdev
    result[:de_vaucouleurs_mixture_weight] = raw_df[:fracdev_r]

    # Note that the SDSS photo pipeline doesn't constrain the de Vaucouleur
    # profile parameters and exponential disk parameters (A/B, angle, scale)
    # to be the same, whereas Celeste does. Here, we pick one or the other
    # from SDSS, based on fracdev - we'll get the parameters corresponding
    # to the dominant component. Later, we limit comparison to objects with
    # fracdev close to 0 or 1 to ensure that we're comparing apples to apples.

    result[:minor_major_axis_ratio] = dev_or_exp(:devab_r, :expab_r)

    # gal effective radius (re)
    re_arcsec = dev_or_exp(:devrad_r, :exprad_r)
    re_pixel = re_arcsec ./ SDSS_ARCSEC_PER_PIXEL
    result[:half_light_radius_px] = convert(Vector{Float64}, re_pixel)

    # gal angle (degrees)
    raw_phi = dev_or_exp(:devphi_r, :expphi_r)
    result[:angle_deg] = raw_phi - floor.(raw_phi / 180) * 180

    return result
end

function object_dict_to_data_frame(objects::Dict)
    result_df = DataFrame()
    for (key, values) in objects
        result_df[Symbol(key)] = values
    end
    result_df
end

"""
Load the SDSS photoObj catalog used to initialize celeste, and reformat column
names to match what the rest of the scoring code expects.
"""
function load_primary(rcf::SDSSIO.RunCamcolField, stagedir::String)
    dir = @sprintf "%s/%d/%d/%d" stagedir rcf.run rcf.camcol rcf.field
    filename = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir rcf.run rcf.camcol rcf.field
    @printf("Loading primary catalog from %s\n", filename)
    raw_df = object_dict_to_data_frame(SDSSIO.read_photoobj(filename))

    usedev = raw_df[:frac_dev] .> 0.5  # true=> use dev, false=> use exp
    dev_or_exp(dev_column, exp_column) = ifelse(usedev, raw_df[dev_column], raw_df[exp_column])
    function star_or_galaxy(star_column, galaxy_dev_column, galaxy_exp_column)
        ifelse(raw_df[:is_star], raw_df[star_column], dev_or_exp(galaxy_dev_column, galaxy_exp_column))
    end

    flux_u = star_or_galaxy(:psfflux_u, :devflux_u, :expflux_u)
    flux_g = star_or_galaxy(:psfflux_g, :devflux_g, :expflux_g)
    flux_r = star_or_galaxy(:psfflux_r, :devflux_r, :expflux_r)
    flux_i = star_or_galaxy(:psfflux_i, :devflux_i, :expflux_i)
    flux_z = star_or_galaxy(:psfflux_z, :devflux_z, :expflux_z)

    result = DataFrame()
    result[:objid] = raw_df[:objid]
    result[:right_ascension_deg] = raw_df[:ra]
    result[:declination_deg] = raw_df[:dec]
    result[:is_star] = raw_df[:is_star]

    result[:reference_band_flux_nmgy] = flux_r

    result[:color_log_ratio_ug] = color_from_fluxes.(flux_u, flux_g)
    result[:color_log_ratio_gr] = color_from_fluxes.(flux_g, flux_r)
    result[:color_log_ratio_ri] = color_from_fluxes.(flux_r, flux_i)
    result[:color_log_ratio_iz] = color_from_fluxes.(flux_i, flux_z)

    result[:de_vaucouleurs_mixture_weight] = raw_df[:frac_dev]
    result[:minor_major_axis_ratio] = dev_or_exp(:ab_dev, :ab_exp)

    # gal effective radius (re)
    re_arcsec = dev_or_exp(:theta_dev, :theta_exp)
    re_pixel = re_arcsec ./ SDSS_ARCSEC_PER_PIXEL
    result[:half_light_radius_px] = convert(Vector{Float64}, re_pixel)

    # gal angle (degrees)
    raw_phi = dev_or_exp(:phi_dev, :phi_exp)
    result[:angle_deg] = raw_phi - floor.(raw_phi / 180) * 180

    return result
end

function get_median_fluxes(variational_params::Vector{Float64}, source_type::Int64)
    ids = Model.ids
    fluxes = Vector{Float64}(5)
    fluxes[3] = exp(variational_params[ids.r1[source_type]])
    fluxes[4] = fluxes[3] * exp(variational_params[ids.c1[3, source_type]])
    fluxes[5] = fluxes[4] * exp(variational_params[ids.c1[4, source_type]])
    fluxes[2] = fluxes[3] / exp(variational_params[ids.c1[2, source_type]])
    fluxes[1] = fluxes[2] / exp(variational_params[ids.c1[1, source_type]])
    fluxes
end

function variational_parameters_to_data_frame_row(variational_params::Vector{Float64})
    ids = Model.ids
    result = DataFrame()
    result[:ra] = variational_params[ids.u[1]]
    result[:dec] = variational_params[ids.u[2]]
    result[:is_star] = variational_params[ids.a[1, 1]]
    result[:de_vaucouleurs_mixture_weight] = variational_params[ids.e_dev]
    result[:minor_major_axis_ratio] = variational_params[ids.e_axis]
    result[:half_light_radius_px] = variational_params[ids.e_scale]
    result[:angle_deg] = variational_params[ids.e_angle]

    fluxes = get_median_fluxes(variational_params, result[1, :is_star] > 0.5 ? 1 : 2)
    result[:reference_band_flux_nmgy] = fluxes[3]
    result[:color_log_ratio_ug] = color_from_fluxes.(fluxes[1], fluxes[2])
    result[:color_log_ratio_gr] = color_from_fluxes.(fluxes[2], fluxes[3])
    result[:color_log_ratio_ri] = color_from_fluxes.(fluxes[3], fluxes[4])
    result[:color_log_ratio_iz] = color_from_fluxes.(fluxes[4], fluxes[5])
    result
end


"""
Convert Celeste results to a dataframe.
"""
function celeste_to_df(results::Vector{ParallelRun.OptimizedSource})
    rows = [variational_parameters_to_data_frame_row(result.vs) for result in results]
    vcat(rows...)
end

# This is the ratio of stars derived from the catalog used the generate the prior; the value 0.99 is
# currently used in inference to account for the extra flexibility of the galaxy model
const PRIOR_PROBABILITY_OF_STAR = 0.28

"""
Draw a random source from the Celeste prior, returning a 1-row data frame.
"""
function draw_source_params(prior, object_id)
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
    color_log_ratios = rand(MixtureModel(color_components, color_mixture_weights))

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

    # Use approximate size of SDSS field in degrees
    right_ascension_deg = rand(Uniform(0, 0.14))
    declination_deg = rand(Uniform(0, 0.22))

    DataFrame(
        objid=object_id,
        right_ascension_deg=right_ascension_deg,
        declination_deg=declination_deg,
        is_star=is_star,
        reference_band_flux_nmgy=reference_band_flux_nmgy,
        color_log_ratio_ug=color_log_ratios[1],
        color_log_ratio_gr=color_log_ratios[2],
        color_log_ratio_ri=color_log_ratios[3],
        color_log_ratio_iz=color_log_ratios[4],
        de_vaucouleurs_mixture_weight=de_vaucouleurs_mixture_weight,
        minor_major_axis_ratio=minor_major_axis_ratio,
        half_light_radius_px=half_light_radius_px,
        angle_deg=angle_deg,
    )
end

"""
Draw sources at random from Celeste prior, returning a catalog DF.
"""
function generate_catalog_from_celeste_prior(num_sources::Int64, seed::Int64)
    srand(seed)
    prior = Model.load_prior()
    vcat(
        [draw_source_params(prior, string(index)) for index in 1:num_sources]...
    )
end

"""
Load/store a catalog DF from/to a CSV.
"""
function read_catalog(csv_file::String)
    @printf("Reading '%s'...\n", csv_file)
    readtable(csv_file)
end
function write_catalog(csv_file::String, catalog_df::DataFrame)
    @printf("Writing '%s'...\n", csv_file)
    writetable(csv_file, catalog_df)
end

## Load a multi-extension FITS imagery file

immutable FitsImage
    pixels::Matrix{Float32}
    header::FITSIO.FITSHeader
    wcs::WCS.WCSTransform
end

function read_fits(filename::String)
    println("Reading '$filename'...")
    if !isfile(filename)
        throw(BenchmarkFitsFileNotFound(filename))
    end

    fits = FITSIO.FITS(filename)
    try
        println("Found $(length(fits)) extensions.")
        map(fits) do extension
            pixels = read(extension)
            header = FITSIO.read_header(extension)
            wcs = WCS.from_header(FITSIO.read_header(fits[1], String))[1]
            FitsImage(pixels, header, wcs)
        end
    finally
        close(fits)
    end
end

## Create Celeste images from FITS imagery

function make_psf(psf_sigma_px)
    alphaBar = [1.; 0.]
    xiBar = [0.; 0.]
    tauBar = [psf_sigma_px^2 0.; 0. psf_sigma_px^2]
    [
        Model.PsfComponent(
            alphaBar[k],
            StaticArrays.SVector{2, Float64}(xiBar),
            StaticArrays.SMatrix{2, 2, Float64, 4}(tauBar)
        )
        for k in 1:2
    ]
end

function make_images(band_extensions::Vector{FitsImage})
    map(enumerate(band_extensions)) do pair
        band_index, extension = pair
        height, width = size(extension.pixels)
        Model.Image(
            height,
            width,
            extension.pixels,
            band_index,
            extension.wcs,
            make_psf(extension.header["CLSIGMA"]),
            0, # SDSS run
            0, # SDSS camcol
            0, # SDSS field
            fill(extension.header["CLSKY"], height, width),
            fill(extension.header["CLIOTA"], height),
            Model.RawPSF(Matrix{Float64}(0, 0), 0, 0, Array{Float64,3}(0, 0, 0)),
        )
    end
end

## Create an initialization catalog for Celeste

function typical_band_relative_intensities(is_star::Bool)
    source_type_index = is_star ? 1 : 2
    prior_parameters::Model.PriorParams = Model.load_prior()
    # Band relative intensities are a mixture of lognormals. Which mixture component has the most
    # weight?
    dominant_component = indmax(prior_parameters.k[:, source_type_index])
    # What are the most typical log relative intensities for that component?
    inter_band_ratios = exp.(
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
    # this is the mode. brightness is log normal.
    exp(prior_parameters.r_μ[source_type_index] - prior_parameters.r_σ²[source_type_index])
end

function make_catalog_entry(
    x_position_world_coords, y_position_world_coords, star_fluxes, galaxy_fluxes
)
    Model.CatalogEntry(
        [x_position_world_coords, y_position_world_coords],
        false, # is_star
        # sample_star_fluxes
        star_fluxes,
        # sample_galaxy_fluxes
        galaxy_fluxes,
        0.1, # gal_frac_dev
        0.7, # gal_ab
        pi / 4, # gal_angle
        4., # gal_scale
        "sample", # objid
        0, # thing_id
    )
end

function get_field(header::FITSIO.FITSHeader, label::String, index::Int64)
    key = @sprintf("%s%03d", label, index)
    if haskey(header, key)
        header[key]
    else
        NA
    end
end

function make_initialization_catalog(catalog::DataFrame)
    position_offset_width = SDSS_ARCSEC_PER_PIXEL / ARCSEC_PER_DEGREE # 1 pixel, in degrees
    star_fluxes = typical_band_relative_intensities(true) .* typical_reference_brightness(true)
    galaxy_fluxes = typical_band_relative_intensities(false) .* typical_reference_brightness(false)
    map(eachrow(catalog)) do row
        position_offset = rand(Uniform(-position_offset_width, position_offset_width), 2)
        make_catalog_entry(
            row[:right_ascension_deg],
            row[:declination_deg],
            star_fluxes,
            galaxy_fluxes,
        )
    end
end

end # module AccuracyBenchmark
