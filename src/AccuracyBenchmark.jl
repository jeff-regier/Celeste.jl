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

    # Convert to "celeste" style results.
    gal_mag_u = dev_or_exp(:devmag_u, :expmag_u)
    gal_mag_g = dev_or_exp(:devmag_g, :expmag_g)
    gal_mag_r = dev_or_exp(:devmag_r, :expmag_r)
    gal_mag_i = dev_or_exp(:devmag_i, :expmag_i)
    gal_mag_z = dev_or_exp(:devmag_z, :expmag_z)

    result = DataFrame()
    result[:objid] = raw_df[:objid]
    result[:ra] = raw_df[:ra]
    result[:dec] = raw_df[:dec]
    result[:is_star] = [x != 0 for x in raw_df[:probpsf]]
    result[:star_mag_r] = raw_df[:psfmag_r]
    result[:gal_mag_r] = gal_mag_r

    # star colors
    result[:star_color_ug] = raw_df[:psfmag_u] .- raw_df[:psfmag_g]
    result[:star_color_gr] = raw_df[:psfmag_g] .- raw_df[:psfmag_r]
    result[:star_color_ri] = raw_df[:psfmag_r] .- raw_df[:psfmag_i]
    result[:star_color_iz] = raw_df[:psfmag_i] .- raw_df[:psfmag_z]

    # gal colors
    result[:gal_color_ug] = gal_mag_u .- gal_mag_g
    result[:gal_color_gr] = gal_mag_g .- gal_mag_r
    result[:gal_color_ri] = gal_mag_r .- gal_mag_i
    result[:gal_color_iz] = gal_mag_i .- gal_mag_z

    # gal shape -- fracdev
    result[:gal_fracdev] = raw_df[:fracdev_r]

    # Note that the SDSS photo pipeline doesn't constrain the de Vaucouleur
    # profile parameters and exponential disk parameters (A/B, angle, scale)
    # to be the same, whereas Celeste does. Here, we pick one or the other
    # from SDSS, based on fracdev - we'll get the parameters corresponding
    # to the dominant component. Later, we limit comparison to objects with
    # fracdev close to 0 or 1 to ensure that we're comparing apples to apples.

    result[:gal_ab] = dev_or_exp(:devab_r, :expab_r)

    # gal effective radius (re)
    re_arcsec = dev_or_exp(:devrad_r, :exprad_r)
    re_pixel = re_arcsec / SDSS_ARCSEC_PER_PIXEL
    result[:gal_scale] = re_pixel

    # gal angle (degrees)
    raw_phi = dev_or_exp(:devphi_r, :expphi_r)
    result[:gal_angle] = raw_phi - floor.(raw_phi / 180) * 180

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
    color_ratios = exp(rand(MixtureModel(color_components, color_mixture_weights)))

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

    position_x = rand(Uniform(0, 1))
    position_y = rand(Uniform(0, 1))

    DataFrame(
        objid=object_id,
        ra=position_x,
        dec=position_y,
        is_star=is_star,
        star_mag_r=flux_to_mag(reference_band_flux_nmgy),
        gal_mag_r=flux_to_mag(reference_band_flux_nmgy),
        star_color_ug=color_ratios[1],
        star_color_gr=color_ratios[2],
        star_color_ri=color_ratios[3],
        star_color_iz=color_ratios[4],
        gal_color_ug=color_ratios[1],
        gal_color_gr=color_ratios[2],
        gal_color_ri=color_ratios[3],
        gal_color_iz=color_ratios[4],
        gal_fracdev=de_vaucouleurs_mixture_weight,
        gal_ab=minor_major_axis_ratio,
        gal_scale=half_light_radius_px,
        gal_angle=angle_deg,
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
