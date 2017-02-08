module AccuracyBenchmark

using DataFrames
using Distributions
import FITSIO

import Celeste.Model

const SDSS_ARCSEC_PER_PIXEL = 0.396

"""
convert between SDSS mags and SDSS flux (nMgy)
"""
mag_to_flux(m::AbstractFloat) = 10.^(0.4 * (22.5 - m))
flux_to_mag(nm::AbstractFloat) = nm > 0 ? 22.5 - 2.5 * log10(nm) : NaN

color_from_fluxes(flux1::AbstractFloat, flux2::AbstractFloat) = log(flux2 / flux1)
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

    # star colors
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

end # module AccuracyBenchmark
