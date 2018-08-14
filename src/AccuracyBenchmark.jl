module AccuracyBenchmark

using DataFrames
using Distributions
import FITSIO
using StaticArrays
import WCS
import CSV

import ..Config
import ..DeterministicVI
import ..Model
import ..ParallelRun
using ..SDSSIO
using ..Coordinates: angular_separation, match_coordinates

const ARCSEC_PER_DEGREE = 3600
const SDSS_ARCSEC_PER_PIXEL = 0.396
const SDSS_DATA_DIR = joinpath(Pkg.dir("Celeste"), "test", "data")
const STRIPE82_RCF = SDSSIO.RunCamcolField(4263, 5, 119)
const COADD_CATALOG_FITS = joinpath(Pkg.dir("Celeste"), "test", "data", "coadd_for_4263_5_119.fit")
const GALAXY_ONLY_COLUMNS = [:gal_frac_dev, :gal_axis_ratio, :gal_radius_px, :gal_angle_deg]

struct BenchmarkFitsFileNotFound <: Exception
    filename::String
end

struct MatchException <: Exception
    msg::String
end

struct MissingColumnsError <: Exception
    missing_columns::Vector{Symbol}
end

################################################################################
# Load/store a catalog DF from/to a CSV.
################################################################################

CATALOG_COLUMNS = Set([
    :ra,
    :dec,
    :is_star,
    :flux_r_nmgy,
    :color_ug,
    :color_gr,
    :color_ri,
    :color_iz,
    :gal_frac_dev,
    :gal_axis_ratio,
    :gal_radius_px,
    :gal_angle_deg,
])

STDERR_COLUMNS = Set([
    :log_flux_r_stderr,
    :color_ug_stderr,
    :color_gr_stderr,
    :color_ri_stderr,
    :color_iz_stderr,
])

BAD_COADD_OBJID = Set([
    # this object is actually multiple objects,
    # see http://legacysurvey.org/viewer/jpeg-cutout/?ra=0.5636&dec=0.4445&zoom=16&layer=decals-dr3
    8647474692482203853,

    # bright neighbor not accounted for,
    # see http://skyserver.sdss.org/dr10/en/tools/chart/navi.aspx?ra=0.551730166101009&dec=0.48085411481715&scale=0.2
    8647474692482203816,

    #  possibly a quasar; update: decals labels this light source a galaxy!
    # see http://legacysurvey.org/viewer?ra=0.5531&dec=0.4530&zoom=16&layer=decals-dr3&sources
    8647474692482204612,

    # Qusar: http://skyserver.sdss.org/dr10/en/get/SpecById.ashx?id=435863481728657408
    8647474692482204147,
])

function assert_columns_are_present(catalog_df::DataFrame, required_columns::Set{Symbol})
    missing_columns = setdiff(required_columns, Set(names(catalog_df)))
    if !isempty(missing_columns)
        throw(MissingColumnsError([missing_columns...]))
    end
end

function read_catalog(csv_file::String)
    catalog_df = CSV.read(csv_file, rows_for_type_detect=100)
    assert_columns_are_present(catalog_df, CATALOG_COLUMNS)
    catalog_df
end

function write_catalog(filename::String, catalog_df::DataFrame;
                       append_hash=false)
    assert_columns_are_present(catalog_df, CATALOG_COLUMNS)

    if append_hash
        # Serialize the data frame into an array of bytes.
        # (CSV.write(::IO, ...) currently broken, so we use a temp file.
        tmp = tempname()
        CSV.write(tmp, catalog_df)
        data = open(tmp) do f
            read(f)
        end
        rm(tmp)

        # Hash the bytes and add the hash string to the filename.
        hash_string = hex(hash(data))[1:10]
        base, extension = splitext(filename)
        filename = @sprintf("%s_%s%s", base, hash_string, extension)

        # Write out the file
        open(filename, "w") do f
            write(f, data)
        end
    else
        CSV.write(filename, catalog_df)
    end

    return filename
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
        missing
    else
        log(flux2 / flux1)
    end
end

function color_from_mags(mags1::AbstractFloat, band1::Int, mags2::AbstractFloat, band2::Int)
    color_from_fluxes(mag_to_flux(mags1, band1), mag_to_flux(mags2, band2))
end

canonical_angle(gal_angle_deg) = gal_angle_deg - floor(gal_angle_deg / 180) * 180

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
    :flags, :is_saturated,
]

function load_fits_table_as_data_frame(filename, extension_index, field_names)
    result_df = DataFrame()
    fits = FITSIO.FITS(filename)
    try
        for key in field_names
            result_df[key] = read(fits[extension_index], string(key))
        end
        return result_df
    finally
        close(fits)
    end
end


function load_stripe82_fits_catalog_as_data_frame(filename, extension_index)
    load_fits_table_as_data_frame(filename, extension_index, STRIPE_82_CATALOG_KEYS)
end

"""
Load Stripe 82 objects into a DataFrame. `fits_filename` should be a FITS file created by running a
CasJobs (skyserver.sdss.org/casjobs/) query on the Stripe82 database. See
https://github.com/jeff-regier/Celeste.jl/wiki/About-SDSS-and-Stripe-82.
"""
function load_coadd_catalog(fits_filename)
    raw_df = load_stripe82_fits_catalog_as_data_frame(fits_filename, 2)

    usedev = raw_df[:fracdev_r] .> 0.5  # true=> use dev, false=> use exp
    dev_or_exp(dev_column, exp_column) = ifelse.(usedev, raw_df[dev_column], raw_df[exp_column])
    is_star = [x != 0 for x in raw_df[:probpsf]]
    function star_or_galaxy(star_column, galaxy_dev_column, galaxy_exp_column)
        ifelse.(is_star, raw_df[star_column], dev_or_exp(galaxy_dev_column, galaxy_exp_column))
    end

    mag_u = star_or_galaxy(:psfmag_u, :devmag_u, :expmag_u)
    mag_g = star_or_galaxy(:psfmag_g, :devmag_g, :expmag_g)
    mag_r = star_or_galaxy(:psfmag_r, :devmag_r, :expmag_r)
    mag_i = star_or_galaxy(:psfmag_i, :devmag_i, :expmag_i)
    mag_z = star_or_galaxy(:psfmag_z, :devmag_z, :expmag_z)

    result = DataFrame()
    result[:objid] = [string(objid) for objid in raw_df[:objid]]
    result[:ra] = raw_df[:ra]
    result[:dec] = raw_df[:dec]
    result[:is_star] = is_star

    flux_r = mag_to_flux.(mag_r, 3)
    result[:flux_r_nmgy] = ifelse.(flux_r .> 0, flux_r, missing)

    result[:color_ug] = color_from_mags.(mag_u, 1, mag_g, 2)
    result[:color_gr] = color_from_mags.(mag_g, 2, mag_r, 3)
    result[:color_ri] = color_from_mags.(mag_r, 3, mag_i, 4)
    result[:color_iz] = color_from_mags.(mag_i, 4, mag_z, 5)

    # gal shape -- fracdev
    result[:gal_frac_dev] = raw_df[:fracdev_r]

    # Note that the SDSS photo pipeline doesn't constrain the de Vaucouleur
    # profile parameters and exponential disk parameters (A/B, angle, scale)
    # to be the same, whereas Celeste does. Here, we pick one or the other
    # from SDSS, based on fracdev - we'll get the parameters corresponding
    # to the dominant component. Later, we limit comparison to objects with
    # fracdev close to 0 or 1 to ensure that we're comparing apples to apples.

    result[:gal_axis_ratio] = dev_or_exp(:devab_r, :expab_r)

    # gal effective radius (re)
    re_arcsec = dev_or_exp(:devrad_r, :exprad_r)
    re_pixel = re_arcsec ./ SDSS_ARCSEC_PER_PIXEL
    result[:gal_radius_px] = convert(Vector{Float64}, re_pixel)

    # gal angle (degrees)
    raw_phi = dev_or_exp(:devphi_r, :expphi_r)
    result[:gal_angle_deg] = canonical_angle.(raw_phi)

    is_saturated = raw_df[:is_saturated] .!= 0
    result = result[.!is_saturated, :]
    bad_rows = [x in BAD_COADD_OBJID for x in result[:, :objid]]
    result = result[.!bad_rows, :]

    # for stars, ensure galaxy-only fields are "missing"
    for col in GALAXY_ONLY_COLUMNS
        result[col] = convert(Vector{Union{Missing, Float64}}, result[col])
        result[result[:is_star], col] = missing
    end

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
    dataset = SDSSDataSet(stagedir)
    raw_df = object_dict_to_data_frame(SDSSIO.read_photoobj(dataset, rcf))

    usedev = raw_df[:frac_dev] .> 0.5  # true=> use dev, false=> use exp
    dev_or_exp(dev_column, exp_column) = ifelse.(usedev, raw_df[dev_column], raw_df[exp_column])
    function star_or_galaxy(star_column, galaxy_dev_column, galaxy_exp_column)
        ifelse.(raw_df[:is_star], raw_df[star_column], dev_or_exp(galaxy_dev_column, galaxy_exp_column))
    end

    flux_u = star_or_galaxy(:psfflux_u, :devflux_u, :expflux_u)
    flux_g = star_or_galaxy(:psfflux_g, :devflux_g, :expflux_g)
    flux_r = star_or_galaxy(:psfflux_r, :devflux_r, :expflux_r)
    flux_i = star_or_galaxy(:psfflux_i, :devflux_i, :expflux_i)
    flux_z = star_or_galaxy(:psfflux_z, :devflux_z, :expflux_z)

    result = DataFrame()
    result[:objid] = raw_df[:objid]
    result[:ra] = raw_df[:ra]
    result[:dec] = raw_df[:dec]
    result[:is_star] = raw_df[:is_star]

    result[:flux_r_nmgy] = flux_r

    result[:color_ug] = color_from_fluxes.(flux_u, flux_g)
    result[:color_gr] = color_from_fluxes.(flux_g, flux_r)
    result[:color_ri] = color_from_fluxes.(flux_r, flux_i)
    result[:color_iz] = color_from_fluxes.(flux_i, flux_z)

    result[:gal_frac_dev] = raw_df[:frac_dev]
    result[:gal_axis_ratio] = dev_or_exp(:ab_dev, :ab_exp)

    # gal effective radius (re)
    re_arcsec = dev_or_exp(:theta_dev, :theta_exp)
    re_pixel = re_arcsec ./ SDSS_ARCSEC_PER_PIXEL
    result[:gal_radius_px] = convert(Vector{Float64}, re_pixel)

    # gal angle (degrees)
    raw_phi = dev_or_exp(:phi_dev, :phi_exp)
    result[:gal_angle_deg] = canonical_angle.(raw_phi)

    # primary is better at indicating oversaturation than coadd
    is_saturated = flux_to_mag.(raw_df[:psfflux_r], 3) .< 16
    result = result[.!is_saturated, :]

    return result
end

function fluxes_from_colors(flux_r_nmgy::Float64, colors::AbstractVector)
    @assert length(colors) == 4
    color_ratios = exp.(colors)
    fluxes = similar(colors, 5)
    fluxes[3] = flux_r_nmgy
    fluxes[4] = fluxes[3] * color_ratios[3]
    fluxes[5] = fluxes[4] * color_ratios[4]
    fluxes[2] = fluxes[3] / color_ratios[2]
    fluxes[1] = fluxes[2] / color_ratios[1]
    fluxes
end

function get_median_fluxes(variational_params::Vector{Float64}, source_type::Int64)
    fluxes_from_colors(
        exp(variational_params[Model.ids.flux_loc[source_type]]),
        variational_params[Model.ids.color_mean[:, source_type]],
    )
end

function variational_parameters_to_data_frame_row(variational_params::Vector{Float64})
    ids = Model.ids
    result = DataFrame()
    result[:ra] = variational_params[ids.pos[1]]
    result[:dec] = variational_params[ids.pos[2]]
    result[:is_star] = variational_params[ids.is_star[1, 1]]
    result[:gal_frac_dev] = variational_params[ids.gal_frac_dev]
    result[:gal_axis_ratio] = variational_params[ids.gal_axis_ratio]
    result[:gal_radius_px] = (
        variational_params[ids.gal_radius_px] * sqrt(variational_params[ids.gal_axis_ratio])
    )
    result[:gal_angle_deg] = canonical_angle(180 / pi * variational_params[ids.gal_angle])

    star_galaxy_index = (result[1, :is_star] > 0.5 ? 1 : 2)
    fluxes = get_median_fluxes(variational_params, star_galaxy_index)
    result[:flux_r_nmgy] = fluxes[3]
    result[:color_ug] = color_from_fluxes(fluxes[1], fluxes[2])
    result[:color_gr] = color_from_fluxes(fluxes[2], fluxes[3])
    result[:color_ri] = color_from_fluxes(fluxes[3], fluxes[4])
    result[:color_iz] = color_from_fluxes(fluxes[4], fluxes[5])

    result[:log_flux_r_stderr] = sqrt(variational_params[ids.flux_scale[star_galaxy_index]])
    result[:color_ug_stderr] = sqrt(variational_params[ids.color_var[1, star_galaxy_index]])
    result[:color_gr_stderr] = sqrt(variational_params[ids.color_var[2, star_galaxy_index]])
    result[:color_ri_stderr] = sqrt(variational_params[ids.color_var[3, star_galaxy_index]])
    result[:color_iz_stderr] = sqrt(variational_params[ids.color_var[4, star_galaxy_index]])

    result
end


"""
Convert Celeste results to a dataframe.
"""
function celeste_to_df(results::Vector{ParallelRun.OptimizedSource})
    rows = []
    for result in results
        if !result.is_sky_bad
            row = variational_parameters_to_data_frame_row(result.vs)
            push!(rows, row)
        end
    end
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
function draw_source_params(prior)
    is_star = (rand(Bernoulli(PRIOR_PROBABILITY_OF_STAR)) == 1)
    source_type_index = is_star ? 1 : 2
    flux_r_nmgy = exp(rand(
        Normal(prior.flux_mean[source_type_index], sqrt(prior.flux_var[source_type_index]))
    ))

    color_mixture_weights = prior.k[:, source_type_index]
    NUM_COLOR_COMPONENTS = length(color_mixture_weights)
    color_components = MvNormal[
        MvNormal(prior.color_mean[:, k, source_type_index], prior.color_cov[:, :, k, source_type_index])
        for k in 1:NUM_COLOR_COMPONENTS
    ]
    colors = rand(MixtureModel(color_components, color_mixture_weights))

    if !is_star
        gal_radius_px = exp(rand(
            Normal(prior.gal_radius_px_mean, sqrt(prior.gal_radius_px_var))
        ))
        gal_angle_deg = rand(Uniform(0, 180))
        gal_axis_ratio = rand(Beta(2, 2))
        gal_frac_dev = rand(Beta(0.5, 0.5))
    else
        gal_radius_px = -1
        gal_angle_deg = -1
        gal_axis_ratio = -1
        gal_frac_dev = -1
    end

    # We should change ra/dec ranges below if we start using another field
    @assert STRIPE82_RCF.run == 4263 && STRIPE82_RCF.camcol == 5 && STRIPE82_RCF.field == 119

    # Use approximate size of SDSS field in degrees
    ra = rand(Uniform(0.443, 0.606))
    dec = rand(Uniform(0.411, 0.635))

    DataFrame(
        ra=ra,
        dec=dec,
        is_star=is_star,
        flux_r_nmgy=flux_r_nmgy,
        color_ug=colors[1],
        color_gr=colors[2],
        color_ri=colors[3],
        color_iz=colors[4],
        gal_frac_dev=gal_frac_dev,
        gal_axis_ratio=gal_axis_ratio,
        gal_radius_px=gal_radius_px,
        gal_angle_deg=gal_angle_deg,
    )
end

"""
Draw sources at random from Celeste prior, returning a catalog DF.
"""
function generate_catalog_from_celeste_prior(num_sources::Int64, seed::Int64)
    srand(seed)
    prior = Model.load_prior()
    result = vcat(
        [draw_source_params(prior) for index in 1:num_sources]...
    )

    # for stars, ensure galaxy-only fields are "missing"
    for col in GALAXY_ONLY_COLUMNS
        result[col] = convert(Vector{Union{Missing, Float64}}, result[col])
        result[result[:is_star], col] = missing
    end

    return result
end


################################################################################
# Support for running Celeste on test imagery
################################################################################

## Load a multi-extension FITS imagery file

struct FitsImage
    pixels::Matrix{Float32}
    header::FITSIO.FITSHeader
    wcs::WCS.WCSTransform
end

function read_fits(filename::String)
    if !isfile(filename)
        throw(BenchmarkFitsFileNotFound(filename))
    end

    fits = FITSIO.FITS(filename)
    try
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

function make_simple_psf(psf_sigma_px)
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

function make_psf_from_header(header::FITSIO.FITSHeader)
    if haskey(header, "CLSIGMA")
        return make_simple_psf(header["CLSIGMA"])
    end

    num_components = header["PSFNCOMP"]
    map(1:num_components) do index
        get(field) = header["PSF$(index)_$(field)"]
        alpha_bar = get("A")
        xi_bar = Float64[get("X1"), get("X2")]
        tau_bar = Float64[get("T11") get("T12"); get("T12") get("T22")]
        Model.PsfComponent(
            alpha_bar,
            StaticArrays.SVector{2, Float64}(xi_bar),
            StaticArrays.SMatrix{2, 2, Float64, 4}(tau_bar),
        )
    end
end

function serialize_psf_to_header(psf::Vector{Model.PsfComponent}, header::FITSIO.FITSHeader)
    header["PSFNCOMP"] = length(psf)
    for (index, component) in enumerate(psf)
        @assert component.tauBar[1, 2] == component.tauBar[2, 1]
        function set(field, value)
            header["PSF$(index)_$(field)"] = value
        end
        set("A", component.alphaBar)
        set("X1", component.xiBar[1])
        set("X2", component.xiBar[2])
        set("T11", component.tauBar[1, 1])
        set("T12", component.tauBar[1, 2])
        set("T22", component.tauBar[2, 2])
    end
end

function make_image(pixels::Matrix{Float32}, band_index::Int,
                    wcs::WCS.WCSTransform, psf::Vector{Model.PsfComponent},
                    sky_level_nmgy::Real, nelec_per_nmgy::Real)
    nx, ny = size(pixels)

    # Render the PSF on a grid, to be used as a (spatially constant) PSF map.
    psfstamp = Model.render_psf(psf, (51, 51))

    Model.Image(
        pixels,
        band_index,
        wcs,
        psf,
        fill(Float32(sky_level_nmgy), nx, ny),
        fill(Float32(nelec_per_nmgy), nx),
        Model.ConstantPSFMap(psfstamp)
    )
end

function make_images(band_extensions::Vector{FitsImage})
    map(enumerate(band_extensions)) do pair
        band_index, extension = pair
        make_image(
            extension.pixels,
            band_index,
            extension.wcs,
            make_psf_from_header(extension.header),
            extension.header["CLSKY"]::Float64,
            extension.header["CLIOTA"]::Float64
        )
    end
end

## Create an initialization catalog for Celeste

function typical_band_fluxes(is_star::Bool)
    source_type_index = is_star ? 1 : 2
    prior_parameters::Model.PriorParams = Model.load_prior()
    # this is the mode. brightness is log normal.
    flux_r = exp(
        prior_parameters.flux_mean[source_type_index] - prior_parameters.flux_var[source_type_index]
    )
    # Band relative intensities are a mixture of lognormals. Which mixture component has the most
    # weight?
    dominant_component = indmax(prior_parameters.k[:, source_type_index])
    # What are the most typical log relative intensities for that component?
    colors = (
        prior_parameters.color_mean[:, dominant_component, source_type_index]
        - diag(prior_parameters.color_cov[:, :, dominant_component, source_type_index])
    )
    fluxes_from_colors(flux_r, colors)
end

function make_catalog_entry(x_position_world_coords::Float64,
                            y_position_world_coords::Float64)
    Model.CatalogEntry(
        [x_position_world_coords, y_position_world_coords],
        false, # is_star
        typical_band_fluxes(true),
        typical_band_fluxes(false),
        0.1, # gal_frac_dev
        0.7, # gal_axis_ratio
        pi / 4, # gal_angle
        4., # gal_radius_px
    )
end

ensure_small_flux(value) = (ismissing(value) || value <= 0) ? 1e-6 : value

na_to_default(value, default) = ismissing(value) ? default : value

function make_catalog_entry(row::DataFrameRow)
    colors = Union{Float64, Missing}[row[:color_ug], row[:color_gr],
                                     row[:color_ri], row[:color_iz]]
    fluxes = fluxes_from_colors(row[:flux_r_nmgy], colors)
    fluxes = ensure_small_flux.(fluxes)
    gal_axis_ratio = na_to_default(row[:gal_axis_ratio], 0.8)
    Model.CatalogEntry(
        [row[:ra], row[:dec]],
        row[:is_star] > 0.5,
        fluxes,
        fluxes,
        na_to_default(row[:gal_frac_dev], 0.5),
        gal_axis_ratio,
        na_to_default(row[:gal_angle_deg], 0.) / 180.0 * pi,
        na_to_default(row[:gal_radius_px] / sqrt(gal_axis_ratio), 2.)
        )
end

function make_initialization_catalog(catalog::DataFrame, use_full_initialzation::Bool)
    position_offset_width = SDSS_ARCSEC_PER_PIXEL / ARCSEC_PER_DEGREE # 1 pixel, in degrees
    map(eachrow(catalog)) do row
        if use_full_initialzation
            make_catalog_entry(row)
        else
            make_catalog_entry(
                row[:ra] + position_offset_width,
                row[:dec] - 0.5 * position_offset_width
            )
        end
    end
end

################################################################################
# Support for generating imagery using Synthetic.jl
################################################################################

struct ImageGeometry
    height_px::Int64
    width_px::Int64
    world_coordinate_origin::Tuple{Float64, Float64} # (ra, dec)
    ra_degrees_per_pixel::Float64
    dec_degrees_per_pixel::Float64
end

function get_image_geometry(catalog_data::DataFrame; field_expand_arcsec=20.0)
    min_ra_deg = minimum(catalog_data[:ra])
    max_ra_deg = maximum(catalog_data[:ra])
    min_dec_deg = minimum(catalog_data[:dec])
    max_dec_deg = maximum(catalog_data[:dec])

    width_arcsec = (max_ra_deg - min_ra_deg) * ARCSEC_PER_DEGREE + 2 * field_expand_arcsec
    height_arcsec = (max_dec_deg - min_dec_deg) * ARCSEC_PER_DEGREE + 2 * field_expand_arcsec
    width_px = convert(Int64, round(width_arcsec / SDSS_ARCSEC_PER_PIXEL))
    height_px = convert(Int64, round(height_arcsec / SDSS_ARCSEC_PER_PIXEL))

    dec_degrees_per_pixel = SDSS_ARCSEC_PER_PIXEL / ARCSEC_PER_DEGREE
    ra_degrees_per_pixel = dec_degrees_per_pixel / cosd(min_dec_deg)

    ImageGeometry(
        height_px,
        width_px,
        (
            min_ra_deg - field_expand_arcsec / ARCSEC_PER_DEGREE,
            min_dec_deg - field_expand_arcsec / ARCSEC_PER_DEGREE,
        ),
        ra_degrees_per_pixel,
        dec_degrees_per_pixel,
    )
end

function make_template_images(
    catalog_data::DataFrame, psf_sigma_px::Float64, band_sky_level_nmgy::Vector{Float64},
    band_nelec_per_nmgy::Vector{Float64}
)
    @assert length(band_sky_level_nmgy) == 5
    @assert length(band_nelec_per_nmgy) == 5
    geometry = get_image_geometry(catalog_data)
    println("  Image dimensions $(geometry.height_px) H x $(geometry.width_px) W px")
    wcs = WCS.WCSTransform(
        2, # dimensions
        # reference pixel coordinates...
        crpix=[1., 1.],
        # ...and corresponding reference world coordinates
        crval=[geometry.world_coordinate_origin[1], geometry.world_coordinate_origin[2]],
        # this WCS is a simple linear transformation
        ctype=["RA---TAN", "DEC--TAN"],
        cunit=["deg", "deg"],
        # these are [du/dx du/dy; dv/dx dv/dy]. (u, v) = world coords, (x, y) = pixel coords.
        pc=[0. geometry.ra_degrees_per_pixel; geometry.dec_degrees_per_pixel 0.],
    )
    map(1:5) do band
        make_image(
            zeros(Float32, (geometry.height_px, geometry.width_px)),
            band,
            wcs,
            make_simple_psf(psf_sigma_px),
            band_sky_level_nmgy[band],
            band_nelec_per_nmgy[band],
        )
    end
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

function save_images_to_fits(filename::String, images::Vector{<:Model.Image})
    println("Writing images to $filename...")
    fits_file = FITSIO.FITS(filename, "w")
    for band_image in images
        header = parse_fits_header_from_string(WCS.to_header(band_image.wcs))
        serialize_psf_to_header(band_image.psf, header)
        header["CLSKY"] = mean(band_image.sky.sky_small) * mean(band_image.sky.calibration)
        FITSIO.set_comment!(header, "CLSKY", "Mean sky background per pixel, nMgy")
        header["CLIOTA"] = mean(band_image.nelec_per_nmgy)
        FITSIO.set_comment!(header, "CLIOTA", "Gain, nelec per nMgy")
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

COLOR_COLUMNS = [:color_ug, :color_gr, :color_ri, :color_iz]

ABSOLUTE_ERROR_COLUMNS = vcat(
    [:gal_frac_dev, :gal_axis_ratio, :gal_radius_px],
    COLOR_COLUMNS,
)

function degrees_to_diff(a, b)
    angle_between = abs.(a - b) .% 180
    min.(angle_between, 180 .- angle_between)
end


"""
Given two results data frame, one containing ground truth (i.e Coadd)
and one containing predictions (i.e., either Primary of Celeste),
compute an a data frame containing each prediction's error.
(It's not an average of the errors, it's each error.)
Let's call the return type of this function an \"error data frame\".
"""
function get_error_df(truth::DataFrame, predicted::DataFrame)
    errors = DataFrame()

    predicted_galaxy = predicted[:is_star] .< .5
    true_galaxy = truth[:is_star] .< .5
    errors[:missed_stars] = ifelse.(.!true_galaxy, predicted_galaxy, missing)
    errors[:missed_galaxies] = ifelse.(true_galaxy, .!predicted_galaxy, missing)

    errors[:position] =
        (ARCSEC_PER_DEGREE / SDSS_ARCSEC_PER_PIXEL) .*
        angular_separation.(truth[:ra],
                            truth[:dec],
                            predicted[:ra],
                            predicted[:dec])

    # compare flux in both mags and nMgy for now
    errors[:flux_r_mag] = abs.(
        flux_to_mag.(truth[:flux_r_nmgy], 3)
        .- flux_to_mag.(predicted[:flux_r_nmgy], 3)
    )
    errors[:flux_r_nmgy] = abs.(
        truth[:flux_r_nmgy] .- predicted[:flux_r_nmgy]
    )
    errors[:gal_angle_deg] = degrees_to_diff(truth[:gal_angle_deg], predicted[:gal_angle_deg])

    for column_symbol in ABSOLUTE_ERROR_COLUMNS
        errors[column_symbol] = abs.(truth[column_symbol] .- predicted[column_symbol])
    end
    for color_column in COLOR_COLUMNS
        # to match up with Stripe82Score, which used differences of mags
        errors[color_column] .*= 2.5 / log(10)
    end

    errors
end

function is_good_row(truth_row::DataFrameRow, error_row::DataFrameRow, column_name::Symbol)
    if ismissing(error_row[column_name]) || isnan(error_row[column_name])
        return false
    elseif !ismissing(truth_row[:gal_radius_px]) && truth_row[:gal_radius_px] > 20
        return false
    end

    if column_name in [:gal_axis_ratio, :gal_radius_px, :gal_angle_deg,
                       :gal_frac_dev]
        has_mixture_weight = !ismissing(truth_row[:gal_frac_dev])
        if has_mixture_weight && (0.05 < truth_row[:gal_frac_dev] < 0.95)
            return false
        end
    end
    if column_name == :gal_angle_deg
        if !ismissing(truth_row[:gal_axis_ratio]) && truth_row[:gal_axis_ratio] > .6
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

function score_column(errors::AbstractVector)
    DataFrame(
        N=length(errors),
        first=mean(errors),
    )
end

function score_column(first_errors::AbstractVector,
                      second_errors::AbstractVector)
    @assert length(first_errors) == length(second_errors)
    scores = score_column(first_errors)
    scores[:second] = mean(second_errors)
    diffs = first_errors .- second_errors
    scores[:diff] = mean(diffs)
    scores[:diff_sd] = std(abs.(diffs)) / sqrt(length(diffs))
    scores
end

function get_scores_df(
    truth::DataFrame, first_errors::DataFrame, second_errors::Nullable{DataFrame}
)
    score_rows = DataFrame[]
    for column_name in names(first_errors)
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


"""
    match_catalogs(truth, predictions)

Return subset of rows in `truth` and `predictions`, such that each entry in
`truth` has a corresponding match in all `predictions`. A "match" is the
closest object within `tol` degrees. The output data frames all have the same
number of rows, and rows correspond to matching objects.
"""
function match_catalogs(truth::DataFrame, prediction_dfs::Vector{DataFrame};
                        tol::Float64=(0.396 / 3600.))

    matched = trues(nrow(truth))
    idxs = Vector{Int}[]
    for prediction in prediction_dfs
        # remove missings before feeding to match_coordinates
        @assert sum(ismissing.(truth[:ra])) == 0
        @assert sum(ismissing.(truth[:dec])) == 0
        @assert sum(ismissing.(prediction[:ra])) == 0
        @assert sum(ismissing.(prediction[:dec])) == 0
        idx, dists = match_coordinates(Vector{Float64}(truth[:ra]),
                                       Vector{Float64}(truth[:dec]),
                                       Vector{Float64}(prediction[:ra]),
                                       Vector{Float64}(prediction[:dec]))
        matched .&= dists .< tol
        push!(idxs, idx)
    end

    # For each row in `truth`, `matched` is now true if and only if all
    # predictions have a close match.
    #
    # `idxs[i]` is same length as `truth` and contains the index in
    # `prediction_dfs[i]` that is the closest match to each row in
    # `truth`.
    matched_truth = truth[matched, :]
    matched_predictions = [prediction[idx[matched], :]
                           for (prediction, idx) in zip(prediction_dfs, idxs)]

    return matched_truth, matched_predictions
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

    valid_rows = (matched_truth[:flux_r_nmgy] .> 0)
    matched_truth = matched_truth[valid_rows, :]
    matched_predictions = matched_predictions[valid_rows, :]

    get_errors(column, map_fn) =
        map_fn.(matched_predictions[column]) .- map_fn.(matched_truth[column])
    get_errors(column) = get_errors(column, x -> x)
    errors = [
        get_errors(:flux_r_nmgy, log),
        get_errors(:color_ug),
        get_errors(:color_gr),
        get_errors(:color_ri),
        get_errors(:color_iz),
    ]
    std_errs = [
        matched_predictions[:log_flux_r_stderr],
        matched_predictions[:color_ug_stderr],
        matched_predictions[:color_gr_stderr],
        matched_predictions[:color_ri_stderr],
        matched_predictions[:color_iz_stderr],
    ]
    names = [:log_flux_r_nmgy, :color_ug, :color_gr,
             :color_ri, :color_iz]

    matched_truth[:log_flux_err] = errors[1]
    matched_truth[:log_flux_stderr] = std_errs[1]
    matched_truth[:flux_r_celeste] = matched_predictions[:flux_r_nmgy]
    CSV.write("stderr.csv", matched_truth)

    mapreduce(vcat, zip(names, errors, std_errs)) do values
        name, error, std_err = values
        DataFrame(name=fill(name, length(error)),
                  error=error,
                  posterior_std_err=std_err)
    end
end

function score_uncertainty(uncertainty_df::DataFrame)
    mapreduce(vcat, groupby(uncertainty_df, :name)) do group_df
        abs_error_sds = abs.(group_df[:error] ./ group_df[:posterior_std_err])
        abs_error_sds = abs_error_sds[.!ismissing.(abs_error_sds)]
        DataFrame(
            field=group_df[1, :name],
            within_half_sd=mean(abs_error_sds .<= 1/2),
            within_1_sd=mean(abs_error_sds .<= 1),
            within_2_sd=mean(abs_error_sds .<= 2),
            within_3_sd=mean(abs_error_sds .<= 3),
        )
    end
end


function plot_image(img)
    xs = img.pixels'

    xs -= 550
    black = minimum(filter(.!isnan, xs))
    xs = map((x)->isnan(x) ? black : x, xs)
    xs = min.(xs, 10_000)
    xs = log.(xs + 100)
    #cutoffs = quantile(xs2[:], 0:0.01:1)
    #xs3 = map(x->findfirst(x .< cutoffs), xs2)
    xs -= log(black + 100)
    xs /= log(10_100)
    return xs
end


end # module AccuracyBenchmark
