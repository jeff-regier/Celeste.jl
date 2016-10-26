import csv
import logging
import math
import os

import astropy.io.fits
import galsim

_logger = logging.getLogger(__name__) 

ARCSEC_PER_PIXEL = 0.75
SHIFT_RADIUS_ARCSEC = ARCSEC_PER_PIXEL
PSF_SIGMA_PIXELS = 1
STAMP_SIZE_PX = 48
COUNTS_PER_NMGY = 1000.0 # a.k.a. "iota" in Celeste

# intensity (flux) relative to third band (= "a" band = reference)
# see GalsimBenchmark.typical_band_relative_intensities()
# these are taken from the current dominant component of the lognormal prior on c_s for stars
STAR_RELATIVE_INTENSITIES = [
    1 / (4.986 * 2.049),
    1 / 2.049,
    1,
    1.350,
    1.350 * 1.184,
]
# these are taken from the current dominant component of the lognormal prior on c_s for galaxies
GALAXY_RELATIVE_INTENSITIES = [
    1 / (2.117 * 2.152),
    1 / 2.152,
    1,
    1.421,
    1.421 * 1.299,
]

RANDOM_SEED = 1234

FITS_COMMENT_PREPEND = 'Celeste: '
TRUTH_HEADER_FIELDS = [
    ('world_center_x', float, 'CL_CENTX', 'X center in world coordinates'),
    ('world_center_y', float, 'CL_CENTY', 'Y center in world coordinates'),
    ('star_or_galaxy', str, 'CL_STGAL', '"star" or "galaxy"?'),
    ('angle_degrees', float, 'CL_ANGLE', 'major axis angle (degrees from x-axis)'),
    ('minor_major_axis_ratio', float, 'CL_RATIO', 'minor/major axis ratio'),
    ('half_light_radius_arcsec', float, 'CL_HLRAD', 'half-light radius (arcsec)'),
    ('reference_band_flux_nmgy', float, 'CL_FLUX', 'reference (=3) band brightness (nMgy)'),
    ('sky_level_nmgy', float, 'CL_SKY', '"epsilon" sky level (nMgy each px)'),
    ('add_noise', bool, 'CL_NOISE', 'was Poisson noise added?'),
    ('comment', str, 'CL_DESCR', 'comment'),
]

def make_basic_galaxy(half_light_radius_arcsec, flux_counts):
    return galsim.Exponential(half_light_radius=half_light_radius_arcsec, flux=flux_counts)

def make_psf(flux):
    return galsim.Gaussian(flux=flux, sigma=PSF_SIGMA_PIXELS * ARCSEC_PER_PIXEL)

def apply_shear(galaxy, angle_degrees, minor_major_axis_ratio):
    return galaxy.shear(q=minor_major_axis_ratio, beta=angle_degrees * galsim.degrees)

def apply_shift(light_source, truth_dict):
    image_center_in_world_coordinates = (STAMP_SIZE_PX + 1) / 2.0 * ARCSEC_PER_PIXEL
    return light_source.shift(
        float(truth_dict['world_center_x']) - image_center_in_world_coordinates,
        float(truth_dict['world_center_y']) - image_center_in_world_coordinates,
    )

def create_galaxy_from_truth_parameters(truth_dict, relative_intensity):
    galaxy = make_basic_galaxy(
        float(truth_dict['half_light_radius_arcsec']),
        float(truth_dict['reference_band_flux_nmgy']) * relative_intensity * COUNTS_PER_NMGY,
    )
    galaxy = apply_shear(
        galaxy,
        float(truth_dict['angle_degrees']),
        float(truth_dict['minor_major_axis_ratio']),
    )
    galaxy = apply_shift(galaxy, truth_dict)
    return galaxy

def add_sky_background(image, sky_level_nmgy):
    image.array[:] = image.array + sky_level_nmgy * COUNTS_PER_NMGY

def add_noise(image, uniform_deviate):
    noise = galsim.PoissonNoise(uniform_deviate)
    image.addNoise(noise)

def read_truth(file_name):
    with open(file_name) as truth_stream:
        reader = csv.DictReader(truth_stream)
        for row in reader:
            yield row

def add_header_to_hdu(hdu, case_index, band_index, truth_dict):
    header = galsim.fits.FitsHeader(hdu.header)
    header['CL_CASEI'] = (case_index + 1, FITS_COMMENT_PREPEND + 'test case index')
    header['CL_BAND'] = (band_index + 1, FITS_COMMENT_PREPEND + 'color band')
    header['CL_IOTA'] = (COUNTS_PER_NMGY, FITS_COMMENT_PREPEND + 'counts per nMgy')
    for csv_field, type_fn, fits_field, comment in TRUTH_HEADER_FIELDS:
        if truth_dict[csv_field]:
            value = type_fn(truth_dict[csv_field])
            header[fits_field] = (value, FITS_COMMENT_PREPEND + comment)

def construct_image_from_truth(truth_dict, band_index, uniform_deviate):
    if truth_dict['star_or_galaxy'] == 'galaxy':
        galaxy = create_galaxy_from_truth_parameters(
            truth_dict,
            GALAXY_RELATIVE_INTENSITIES[band_index],
        )
        final_light_source = galsim.Convolve([galaxy, make_psf(1)])
    else:
        assert truth_dict['star_or_galaxy'] == 'star'
        point_source_plus_psf = make_psf(
            float(truth_dict['reference_band_flux_nmgy'])
                * STAR_RELATIVE_INTENSITIES[band_index] * COUNTS_PER_NMGY
        )
        final_light_source = apply_shift(point_source_plus_psf, truth_dict)

    image = galsim.ImageF(STAMP_SIZE_PX, STAMP_SIZE_PX, scale=ARCSEC_PER_PIXEL)
    final_light_source.drawImage(image)
    add_sky_background(image, float(truth_dict['sky_level_nmgy']))
    if truth_dict['add_noise'] == '1':
        add_noise(image, uniform_deviate)
    return image

def save_multi_extension_fits(hdu_list, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))
    if os.path.exists(filename):
        os.remove(filename)
    hdu_list.writeto(filename)
    _logger.info('Wrote multi-extension FITS images to %r', filename)

def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    truth_file_name = 'galsim_truth.csv'
    _logger.info('Reading %s', truth_file_name)
    fits_hdus = astropy.io.fits.HDUList()
    for case_index, truth_dict in enumerate(read_truth(truth_file_name)):
        uniform_deviate = galsim.UniformDeviate(RANDOM_SEED + case_index)
        for band_index in xrange(5):
            image = construct_image_from_truth(truth_dict, band_index, uniform_deviate)
            galsim.fits.write(image, hdu_list=fits_hdus)
            add_header_to_hdu(fits_hdus[-1], case_index, band_index, truth_dict)

    image_file_name = os.path.join('output', 'galsim_test_images.fits')
    save_multi_extension_fits(fits_hdus, image_file_name)

if __name__ == "__main__":
    main()
