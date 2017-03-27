module AccuracyBenchmark

using DataFrames
using Distributions
import FITSIO
import StaticArrays
import WCS

import Celeste.Configs
import Celeste.DeterministicVI
import Celeste.Infer
import Celeste.Model
import Celeste.ParallelRun
import Celeste.SDSSIO

const ARCSEC_PER_DEGREE = 3600
const SDSS_ARCSEC_PER_PIXEL = 0.396
const SDSS_DATA_DIR = joinpath(Pkg.dir("Celeste"), "test", "data")
const STRIPE82_RCF = SDSSIO.RunCamcolField(4263, 5, 119)

immutable BenchmarkFitsFileNotFound <: Exception
    filename::String
end

immutable MatchException <: Exception
    msg::String
end

immutable ObjectsMissingFromGroundTruth <: Exception
    missing_objids::Vector{String}
end

immutable MissingColumnsError <: Exception
    missing_columns::Vector{Symbol}
end

################################################################################
# Load/store a catalog DF from/to a CSV.
################################################################################

CATALOG_COLUMNS = Set([
    :objid,
    :right_ascension_deg,
    :declination_deg,
    :is_star,
    :reference_band_flux_nmgy,
    :color_log_ratio_ug,
    :color_log_ratio_gr,
    :color_log_ratio_ri,
    :color_log_ratio_iz,
    :de_vaucouleurs_mixture_weight,
    :minor_major_axis_ratio,
    :half_light_radius_px,
    :angle_deg,
])

STDERR_COLUMNS = Set([
    :log_reference_band_flux_stderr,
    :color_log_ratio_ug_stderr,
    :color_log_ratio_gr_stderr,
    :color_log_ratio_ri_stderr,
    :color_log_ratio_iz_stderr,
])

function assert_columns_are_present(catalog_df::DataFrame, required_columns::Set{Symbol})
    missing_columns = setdiff(required_columns, Set(names(catalog_df)))
    if !isempty(missing_columns)
        throw(MissingColumnsError([missing_columns...]))
    end
end

function read_catalog(csv_file::String)
    @printf("Reading '%s'...\n", csv_file)
    catalog_df = readtable(csv_file)
    assert_columns_are_present(catalog_df, CATALOG_COLUMNS)
    catalog_df[:objid] = String[string(objid) for objid in catalog_df[:objid]]
    catalog_df
end

function write_catalog(csv_file::String, catalog_df::DataFrame)
    assert_columns_are_present(catalog_df, CATALOG_COLUMNS)
    @printf("Writing '%s'...\n", csv_file)
    writetable(csv_file, catalog_df)
end

function append_hash_to_file(filename::String)
    contents_hash = open(filename) do stream
        hash(read(stream))
    end
    hash_string = hex(contents_hash)[1:10]
    base, extension = splitext(filename)
    new_filename = @sprintf("%s_%s%s", base, hash_string, extension)
    @printf("Renaming %s -> %s\n", filename, new_filename)
    mv(filename, new_filename, remove_destination=true)
end

################################################################################
# Read various catalogs to a common catalog DF format
################################################################################

"""
Convert between SDSS asinh mags and SDSS flux (nMgy).
See http://www.sdss.org/dr12/algorithms/magnitudes/#asinh.
"""
const ASINH_SOFTENING_PARAMETERS = [
    1.4e-10, # for band 1 = u
    0.9e-10,
    1.2e-10,
    1.8e-10,
    7.4e-10, # for band 5 = z
]

function mag_to_flux(mags::AbstractFloat, band_index::Int)
    b = ASINH_SOFTENING_PARAMETERS[band_index]
    1e9 * 2 * b * sinh(-log(10) / 2.5 * mags - log(b))
end

function flux_to_mag(flux_nmgy::AbstractFloat, band_index::Int)
    b = ASINH_SOFTENING_PARAMETERS[band_index]
    -2.5 / log(10) * (asinh(flux_nmgy * 1e-9 / (2*b)) + log(b))
end

function color_from_fluxes(flux1::AbstractFloat, flux2::AbstractFloat)
    if flux1 <= 0 || flux2 <= 0
        NA
    else
        log(flux2 / flux1)
    end
end

function color_from_mags(mags1::AbstractFloat, band1::Int, mags2::AbstractFloat, band2::Int)
    color_from_fluxes(mag_to_flux(mags1, band1), mag_to_flux(mags2, band2))
end

canonical_angle(angle_deg) = angle_deg - floor(angle_deg / 180) * 180

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
    result[:objid] = [string(objid) for objid in raw_df[:objid]]
    result[:right_ascension_deg] = raw_df[:ra]
    result[:declination_deg] = raw_df[:dec]
    result[:is_star] = is_star

    result[:reference_band_flux_nmgy] = mag_to_flux.(mag_r, 3)

    result[:color_log_ratio_ug] = color_from_mags.(mag_u, 1, mag_g, 2)
    result[:color_log_ratio_gr] = color_from_mags.(mag_g, 2, mag_r, 3)
    result[:color_log_ratio_ri] = color_from_mags.(mag_r, 3, mag_i, 4)
    result[:color_log_ratio_iz] = color_from_mags.(mag_i, 4, mag_z, 5)

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
    result[:angle_deg] = canonical_angle.(raw_phi)

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
    result[:angle_deg] = canonical_angle.(raw_phi)

    # primary is better at flagging oversaturated sources than coadd
    result = result[flux_to_mag.(raw_df[:psfflux_r], 3) .>= 16, :]

    return result
end

function fluxes_from_colors(reference_band_flux_nmgy::Float64, color_log_ratios::DataVector{Float64})
    @assert length(color_log_ratios) == 4
    color_ratios = exp.(color_log_ratios)
    fluxes = DataArray(Float64, 5)
    fluxes[3] = reference_band_flux_nmgy
    fluxes[4] = fluxes[3] * color_ratios[3]
    fluxes[5] = fluxes[4] * color_ratios[4]
    fluxes[2] = fluxes[3] / color_ratios[2]
    fluxes[1] = fluxes[2] / color_ratios[1]
    fluxes
end

function fluxes_from_colors(reference_band_flux_nmgy::Float64, color_log_ratios::Vector{Float64})
    fluxes = fluxes_from_colors(reference_band_flux_nmgy, DataArray{Float64}(color_log_ratios))
    convert(Vector{Float64}, fluxes)
end

function get_median_fluxes(variational_params::Vector{Float64}, source_type::Int64)
    fluxes_from_colors(
        exp(variational_params[Model.ids.r1[source_type]]),
        variational_params[Model.ids.c1[:, source_type]],
    )
end

function variational_parameters_to_data_frame_row(objid::String, variational_params::Vector{Float64})
    ids = Model.ids
    result = DataFrame()
    result[:objid] = objid
    result[:right_ascension_deg] = variational_params[ids.u[1]]
    result[:declination_deg] = variational_params[ids.u[2]]
    result[:is_star] = variational_params[ids.a[1, 1]]
    result[:de_vaucouleurs_mixture_weight] = variational_params[ids.e_dev]
    result[:minor_major_axis_ratio] = variational_params[ids.e_axis]
    result[:half_light_radius_px] = (
        variational_params[ids.e_scale] * sqrt(variational_params[ids.e_axis])
    )
    result[:angle_deg] = canonical_angle(180 / pi * variational_params[ids.e_angle])

    star_galaxy_index = (result[1, :is_star] > 0.5 ? 1 : 2)
    fluxes = get_median_fluxes(variational_params, star_galaxy_index)
    result[:reference_band_flux_nmgy] = fluxes[3]
    result[:color_log_ratio_ug] = color_from_fluxes(fluxes[1], fluxes[2])
    result[:color_log_ratio_gr] = color_from_fluxes(fluxes[2], fluxes[3])
    result[:color_log_ratio_ri] = color_from_fluxes(fluxes[3], fluxes[4])
    result[:color_log_ratio_iz] = color_from_fluxes(fluxes[4], fluxes[5])

    result[:log_reference_band_flux_stderr] = sqrt(variational_params[ids.r2[star_galaxy_index]])
    result[:color_log_ratio_ug_stderr] = sqrt(variational_params[ids.c2[1, star_galaxy_index]])
    result[:color_log_ratio_gr_stderr] = sqrt(variational_params[ids.c2[2, star_galaxy_index]])
    result[:color_log_ratio_ri_stderr] = sqrt(variational_params[ids.c2[3, star_galaxy_index]])
    result[:color_log_ratio_iz_stderr] = sqrt(variational_params[ids.c2[4, star_galaxy_index]])

    result
end


"""
Convert Celeste results to a dataframe.
"""
function celeste_to_df(results::Vector{ParallelRun.OptimizedSource})
    rows = [variational_parameters_to_data_frame_row(result.objid, result.vs) for result in results]
    vcat(rows...)
end

################################################################################
# Generate a random catalog from the Celeste prior
################################################################################

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


################################################################################
# Support for running Celeste on test imagery
################################################################################

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

function make_image(
    pixels::Matrix{Float32}, band_index::Int, wcs::WCS.WCSTransform, psf_sigma_px::Float64,
    sky_level_nmgy::Float64, counts_per_nmgy::Float64
)
    height_px, width_px = size(pixels)
    sky_intensity = Model.SkyIntensity(
        fill(sky_level_nmgy, height_px, width_px),
        collect(1:height_px),
        collect(1:width_px),
        ones(height_px),
    )
    iota_vec = fill(counts_per_nmgy, height_px)
    Model.Image(
        height_px,
        width_px,
        pixels,
        band_index,
        wcs,
        make_psf(psf_sigma_px),
        0, 0, 0, # run, camcol, field
        sky_intensity,
        iota_vec,
        Model.RawPSF(Matrix{Float64}(0, 0), 0, 0, Array{Float64,3}(0, 0, 0)),
    )
end

function make_images(band_extensions::Vector{FitsImage})
    map(enumerate(band_extensions)) do pair
        band_index, extension = pair
        make_image(
            extension.pixels,
            band_index,
            extension.wcs,
            convert(Float64, extension.header["CLSIGMA"]),
            convert(Float64, extension.header["CLSKY"]),
            convert(Float64, extension.header["CLIOTA"]),
        )
    end
end

## Create an initialization catalog for Celeste

function typical_band_fluxes(is_star::Bool)
    source_type_index = is_star ? 1 : 2
    prior_parameters::Model.PriorParams = Model.load_prior()
    # this is the mode. brightness is log normal.
    reference_band_flux = exp(
        prior_parameters.r_μ[source_type_index] - prior_parameters.r_σ²[source_type_index]
    )
    # Band relative intensities are a mixture of lognormals. Which mixture component has the most
    # weight?
    dominant_component = indmax(prior_parameters.k[:, source_type_index])
    # What are the most typical log relative intensities for that component?
    color_log_ratios = (
        prior_parameters.c_mean[:, dominant_component, source_type_index]
        - diag(prior_parameters.c_cov[:, :, dominant_component, source_type_index])
    )
    fluxes_from_colors(reference_band_flux, color_log_ratios)
end

function make_catalog_entry(
    x_position_world_coords::Float64, y_position_world_coords::Float64, objid::String
)
    Model.CatalogEntry(
        [x_position_world_coords, y_position_world_coords],
        false, # is_star
        typical_band_fluxes(true),
        typical_band_fluxes(false),
        0.1, # gal_frac_dev
        0.7, # gal_ab
        pi / 4, # gal_angle
        4., # gal_scale
        objid, # objid
        0, # thing_id
    )
end

ensure_small_flux(value) = (isna(value) || value <= 0) ? 1e-6 : value

na_to_default(value, default) = isna(value) ? default : value

function make_catalog_entry(row::DataFrameRow)
    color_log_ratios = DataArray{Float64}(DataArray(Any[
        row[:color_log_ratio_ug],
        row[:color_log_ratio_gr],
        row[:color_log_ratio_ri],
        row[:color_log_ratio_iz],
    ]))
    fluxes = fluxes_from_colors(row[:reference_band_flux_nmgy], color_log_ratios)
    fluxes = convert(Vector{Float64}, ensure_small_flux.(fluxes))
    Model.CatalogEntry(
        [row[:right_ascension_deg], row[:declination_deg]],
        row[:is_star],
        fluxes,
        fluxes,
        na_to_default(row[:de_vaucouleurs_mixture_weight], 0.),
        na_to_default(row[:minor_major_axis_ratio], 0.),
        na_to_default(row[:angle_deg], 0.) / 180.0 * pi,
        na_to_default(row[:half_light_radius_px], 0.),
        row[:objid],
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

function make_initialization_catalog(catalog::DataFrame, use_full_initialzation::Bool)
    position_offset_width = SDSS_ARCSEC_PER_PIXEL / ARCSEC_PER_DEGREE # 1 pixel, in degrees
    map(eachrow(catalog)) do row
        if use_full_initialzation
            make_catalog_entry(row)
        else
            make_catalog_entry(
                row[:right_ascension_deg] + position_offset_width,
                row[:declination_deg] - 0.5 * position_offset_width,
                row[:objid],
            )
        end
    end
end

################################################################################
# Support for generating imagery using Synthetic.jl
################################################################################

immutable ImageGeometry
    height_px::Int64
    width_px::Int64
    world_coordinate_origin::Tuple{Float64, Float64}
end

function get_image_geometry(catalog_data::DataFrame; field_expand_arcsec=20.0)
    min_ra_deg = minimum(catalog_data[:right_ascension_deg])
    max_ra_deg = maximum(catalog_data[:right_ascension_deg])
    min_dec_deg = minimum(catalog_data[:declination_deg])
    max_dec_deg = maximum(catalog_data[:declination_deg])

    width_arcsec = (max_ra_deg - min_ra_deg) * ARCSEC_PER_DEGREE + 2 * field_expand_arcsec
    height_arcsec = (max_dec_deg - min_dec_deg) * ARCSEC_PER_DEGREE + 2 * field_expand_arcsec
    width_px = convert(Int64, round(width_arcsec / SDSS_ARCSEC_PER_PIXEL))
    height_px = convert(Int64, round(height_arcsec / SDSS_ARCSEC_PER_PIXEL))

    ImageGeometry(
        height_px,
        width_px,
        (
            min_ra_deg - field_expand_arcsec / ARCSEC_PER_DEGREE,
            min_dec_deg - field_expand_arcsec / ARCSEC_PER_DEGREE,
        )
    )
end

function make_template_images(
    catalog_data::DataFrame, psf_sigma_px::Float64, sky_level_nmgy::Float64,
    counts_per_nmgy::Float64
)
    geometry = get_image_geometry(catalog_data)
    println("  Image dimensions $(geometry.height_px) H x $(geometry.width_px) W px")
    dec_deg_per_pixel = SDSS_ARCSEC_PER_PIXEL / ARCSEC_PER_DEGREE
    ra_deg_per_pixel = dec_deg_per_pixel / cosd(geometry.world_coordinate_origin[2])
    wcs = WCS.WCSTransform(
        2, # dimensions
        # reference pixel coordinates...
        crpix=[0., 0.],
        # ...and corresponding reference world coordinates
        crval=[geometry.world_coordinate_origin[1], geometry.world_coordinate_origin[2]],
        # this WCS is a simple linear transformation
        ctype=["RA---TAN", "DEC--TAN"],
        cunit=["deg", "deg"],
        # these are [du/dx du/dy; dv/dx dv/dy]. (u, v) = world coords, (x, y) = pixel coords.
        pc=[0. ra_deg_per_pixel; dec_deg_per_pixel 0.],
    )
    map(1:5) do band
        make_image(
            zeros(Float32, (geometry.height_px, geometry.width_px)),
            band,
            wcs,
            psf_sigma_px,
            sky_level_nmgy,
            counts_per_nmgy,
        )
    end
end

function extract_psf_sigma_px(psf::Vector{Model.PsfComponent})
    # ensure the PSF is a multiple of identity covariance
    @assert psf[1].alphaBar == 1.
    @assert all(psf[1].xiBar .== [0.; 0.])
    @assert psf[1].tauBar[1, 2] == 0.
    @assert psf[1].tauBar[2, 1] == 0.
    @assert psf[1].tauBar[1, 1] == psf[1].tauBar[2, 2]
    sqrt(psf[1].tauBar[1, 1])
end

## Parse a FITS header from a raw string representation

# It's silly that we have to do this, but the Julia FITSIO library doesn't support this (it can only
# read a header from a file), while the Julia WCS library will only generate a header in raw string
# form. (The Julia FITSIO library also won't write a raw string header to a file, so scrap that idea
# :).) Fortunately, the FITS header format is a simple fixed-width textual format with only a few
# data types. See https://fits.gsfc.nasa.gov/fits_primer.html.

function parse_fits_header_value(raw_value::AbstractString)
    if raw_value == "T"
        true
    elseif raw_value == "F"
        false
    elseif raw_value[1] == '\''
        convert(String, raw_value[2:(end - 1)])
    elseif ismatch(r"^[0-9]+$", raw_value)
        parse(Int64, raw_value)
    else
        parse(Float64, raw_value)
    end
end

function parse_fits_header_from_string(header_string::AbstractString)
    header = FITSIO.FITSHeader(String[], [], String[])
    start_position = 1
    while start_position < length(header_string)
        field_string = header_string[start_position:(start_position + 79)]
        field_name = strip(field_string[1:8])
        if length(field_name) != 0 && field_string[9:10] == "= "
            field_parts = split(field_string[11:80], "/", limit=2)
            raw_value = field_parts[1]
            header[field_name] = parse_fits_header_value(strip(raw_value))
            if length(field_parts) == 2
                comment = strip(field_parts[2])
                if length(comment) > 0
                    FITSIO.set_comment!(header, field_name, convert(String, comment))
                end
            end
        end
        start_position += 80
    end
    header
end

function save_images_to_fits(filename::String, images::Vector{Model.Image})
    println("Writing images to $filename...")
    fits_file = FITSIO.FITS(filename, "w")
    for band_image in images
        header = parse_fits_header_from_string(WCS.to_header(band_image.wcs))
        header["CLSKY"] = band_image.sky.sky_small[1, 1]
        header["CLSIGMA"] = extract_psf_sigma_px(band_image.psf)
        header["CLIOTA"] = band_image.iota_vec[1]
        write(
            fits_file,
            band_image.pixels,
            header=header,
            name="CELESTE_FIELD_$(band_image.b)",
        )
    end
    close(fits_file)
end

################################################################################
# Score a set of predictions against a ground truth catalog
################################################################################

COLOR_COLUMNS = [:color_log_ratio_ug, :color_log_ratio_gr, :color_log_ratio_ri, :color_log_ratio_iz]

ABSOLUTE_ERROR_COLUMNS = vcat(
    [:de_vaucouleurs_mixture_weight, :minor_major_axis_ratio, :half_light_radius_px],
    COLOR_COLUMNS,
)

function degrees_to_diff(a, b)
    angle_between = abs(a - b) % 180
    min.(angle_between, 180 - angle_between)
end

"""
Given two results data frame, one containing ground truth (i.e Coadd)
and one containing predictions (i.e., either Primary of Celeste),
compute an a data frame containing each prediction's error.
(It's not an average of the errors, it's each error.)
Let's call the return type of this function an \"error data frame\".
"""
function get_error_df(truth::DataFrame, predicted::DataFrame)
    errors = DataFrame(objid=truth[:objid])

    predicted_galaxy = predicted[:is_star] .< .5
    true_galaxy = truth[:is_star] .< .5
    errors[:missed_stars] = !true_galaxy .& predicted_galaxy
    errors[:missed_galaxies] = true_galaxy .& !predicted_galaxy

    errors[:position] = sky_distance_px.(
        truth[:right_ascension_deg],
        truth[:declination_deg],
        predicted[:right_ascension_deg],
        predicted[:declination_deg],
    )

    # compare flux in both mags and nMgy for now
    errors[:reference_band_flux_mag] = abs(
        flux_to_mag.(truth[:reference_band_flux_nmgy], 3)
        .- flux_to_mag.(predicted[:reference_band_flux_nmgy], 3)
    )
    errors[:reference_band_flux_nmgy] = abs(
        truth[:reference_band_flux_nmgy] .- predicted[:reference_band_flux_nmgy]
    )
    errors[:angle_deg] = degrees_to_diff(truth[:angle_deg], predicted[:angle_deg])

    for column_symbol in ABSOLUTE_ERROR_COLUMNS
        errors[column_symbol] = abs(truth[column_symbol] - predicted[column_symbol])
    end
    for color_column in COLOR_COLUMNS
        # to match up with Stripe82Score, which used differences of mags
        errors[color_column] *= 2.5 / log(10)
    end

    errors
end

function is_good_row(truth_row::DataFrameRow, error_row::DataFrameRow, column_name::Symbol)
    if isna(error_row[column_name]) || isnan(error_row[column_name])
        return false
    elseif !isna(truth_row[:half_light_radius_px]) && truth_row[:half_light_radius_px] > 20
        return false
    end

    if column_name in [:minor_major_axis_ratio, :half_light_radius_px, :angle_deg,
                       :de_vaucouleurs_mixture_weight]
        has_mixture_weight = !isna(truth_row[:de_vaucouleurs_mixture_weight])
        if has_mixture_weight && (0.05 < truth_row[:de_vaucouleurs_mixture_weight] < 0.95)
            return false
        end
    end
    if column_name == :angle_deg
        if !isna(truth_row[:minor_major_axis_ratio]) && truth_row[:minor_major_axis_ratio] > .6
            return false
        end
    end
    return true
end

function filter_rows(truth::DataFrame, errors::DataFrame, column_name::Symbol)
    map(zip(eachrow(truth), eachrow(errors))) do rows
        (truth_row, error_row) = rows
        is_good_row(truth_row, error_row, column_name)
    end
end

function score_column(errors::DataArray)
    DataFrame(
        N=length(errors),
        first=mean(errors),
    )
end

function score_column(first_errors::DataArray, second_errors::DataArray)
    @assert length(first_errors) == length(second_errors)
    scores = score_column(first_errors)
    scores[:second] = mean(second_errors)
    diffs = first_errors .- second_errors
    scores[:diff] = mean(diffs)
    scores[:diff_sd] = std(abs(diffs)) / sqrt(length(diffs))
    scores
end

function get_scores_df(
    truth::DataFrame, first_errors::DataFrame, second_errors::Nullable{DataFrame}
)
    score_rows = DataFrame[]
    for column_name in names(first_errors)
        if column_name == :objid
            continue
        end

        good_row = filter_rows(truth, first_errors, column_name)
        if !isnull(second_errors)
            good_row .&= filter_rows(truth, get(second_errors), column_name)
        end
        if sum(good_row) <= 1
            continue
        else
            if isnull(second_errors)
                row = score_column(first_errors[good_row, column_name])
            else
                row = score_column(
                    first_errors[good_row, column_name],
                    get(second_errors)[good_row, column_name],
                )
            end

            row[:field] = column_name
            push!(score_rows, row)
        end
    end
    vcat(score_rows...)
end

function extract_rows_by_objid(catalog_df::DataFrame, objids::Vector{String})
    row_indices = map(objids) do objid
        findfirst(catalog_df[:objid] .== objid)
    end
    @assert all(row_indices .!= 0)
    catalog_df[row_indices, :]
end

function match_catalogs(truth::DataFrame, prediction_dfs::Vector{DataFrame})
    objid_sets = [Set(predictions[:objid]) for predictions in prediction_dfs]
    common_objids = intersect(objid_sets...)
    @printf("%d objids in common\n", length(common_objids))

    objids_not_in_ground_truth = setdiff(common_objids, truth[:objid])
    if !isempty(objids_not_in_ground_truth)
        @printf("%d objids not found in ground truth catalog:\n", length(objids_not_in_ground_truth))
        println(join(objids_not_in_ground_truth, ", "))
        throw(ObjectsMissingFromGroundTruth([objids_not_in_ground_truth]))
    end

    ordered_objids = [common_objids...]
    (
        extract_rows_by_objid(truth, ordered_objids),
        [extract_rows_by_objid(predictions, ordered_objids) for predictions in prediction_dfs]
    )
end

function score_predictions(truth::DataFrame, prediction_dfs::Vector{DataFrame})
    matched_truth, matched_prediction_dfs = match_catalogs(truth, prediction_dfs)
    error_dfs = [get_error_df(matched_truth, predictions) for predictions in matched_prediction_dfs]
    @assert length(prediction_dfs) <= 2
    get_scores_df(
        matched_truth,
        error_dfs[1],
        length(error_dfs) > 1 ? Nullable(error_dfs[2]) : Nullable{DataFrame}(),
    )
end

function get_uncertainty_df(truth::DataFrame, predictions::DataFrame)
    assert_columns_are_present(predictions, STDERR_COLUMNS)
    matched_truth, matched_prediction_dfs = match_catalogs(truth, [predictions])
    matched_predictions = matched_prediction_dfs[1]

    valid_rows = (matched_truth[:reference_band_flux_nmgy] .> 0)
    matched_truth = matched_truth[valid_rows, :]
    matched_predictions = matched_predictions[valid_rows, :]

    get_errors(column, map_fn) =
        map_fn.(matched_predictions[column]) .- map_fn.(matched_truth[column])
    get_errors(column) = get_errors(column, x -> x)
    errors = [
        get_errors(:reference_band_flux_nmgy, log),
        get_errors(:color_log_ratio_ug),
        get_errors(:color_log_ratio_gr),
        get_errors(:color_log_ratio_ri),
        get_errors(:color_log_ratio_iz),
    ]
    std_errs = [
        matched_predictions[:log_reference_band_flux_stderr],
        matched_predictions[:color_log_ratio_ug_stderr],
        matched_predictions[:color_log_ratio_gr_stderr],
        matched_predictions[:color_log_ratio_ri_stderr],
        matched_predictions[:color_log_ratio_iz_stderr],
    ]
    names = [:log_reference_band_flux_nmgy, :color_log_ratio_ug, :color_log_ratio_gr,
             :color_log_ratio_ri, :color_log_ratio_iz]

    mapreduce(vcat, zip(names, errors, std_errs)) do values
        name, error, std_err = values
        DataFrame(
            objid=matched_truth[:objid],
            name=fill(name, length(error)),
            error=error,
            posterior_std_err=std_err,
        )
    end
end

function score_uncertainty(uncertainty_df::DataFrame)
    mapreduce(vcat, groupby(uncertainty_df, :name)) do group_df
        abs_error_sds = abs.(group_df[:error] ./ group_df[:posterior_std_err])
        abs_error_sds = abs_error_sds[!isna(abs_error_sds)]
        DataFrame(
            field=group_df[1, :name],
            within_half_sd=mean(abs_error_sds .<= 1/2),
            within_1_sd=mean(abs_error_sds .<= 1),
            within_2_sd=mean(abs_error_sds .<= 2),
            within_3_sd=mean(abs_error_sds .<= 3),
        )
    end
end

################################################################################
# Utilities
################################################################################

"""
Return distance in pixels using small-distance approximation. Falls apart at poles and RA boundary.
"""
function sky_distance_px(ra1, dec1, ra2, dec2)
    distance_deg = sqrt((dec2 - dec1)^2 + (cosd(dec1) * (ra2 - ra1))^2)
    distance_deg * ARCSEC_PER_DEGREE / SDSS_ARCSEC_PER_PIXEL
end

"""
match_position(ras, decs, ra, dec, dist)

Return index of first position in `ras`, `decs` that is within a distance `maxdist_px` of the target
position `ra`, `dec`. If none found, an exception is raised.
"""
function match_position(ras, decs, ra, dec, maxdist_px)
    @assert length(ras) == length(decs)
    filter(eachindex(ras)) do index
        sky_distance_px(ra, dec, ras[index], decs[index]) < maxdist_px
    end
end

# Run Celeste with any combination of single/joint inference
function run_celeste(
    config::Configs.Config, catalog_entries, target_sources, images;
    use_joint_inference=false,
)
    neighbor_map = Infer.find_neighbors(target_sources, catalog_entries, images)
    if use_joint_inference
        ParallelRun.one_node_joint_infer(
            config,
            catalog_entries,
            target_sources,
            neighbor_map,
            images,
        )
    else
        ParallelRun.one_node_single_infer(
            config,
            catalog_entries,
            target_sources,
            neighbor_map,
            images,
        )
    end
end

end # module AccuracyBenchmark
