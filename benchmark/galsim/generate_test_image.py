import csv
import logging
import math
import os

import galsim

_logger = logging.getLogger(__name__) 

ARCSEC_PER_PIXEL = 0.75 # arcsec / pixel
SHIFT_RADIUS_ARCSEC = ARCSEC_PER_PIXEL
PSF_SIGMA_PIXELS = 1
STAMP_SIZE_PX = 48
COUNTS_PER_NMGY = 1000 # a.k.a. "iota" in Celeste

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


def add_noise(image, uniform_deviate):
    noise = galsim.PoissonNoise(uniform_deviate)
    image.addNoise(noise)

def read_truth(file_name):
    with open(file_name) as truth_stream:
        reader = csv.DictReader(truth_stream)
        for row in reader:
            yield row

def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    truth_file_name = 'galsim_truth.csv'
    _logger.info('Reading %s', truth_file_name)
    images = []
    for index, truth_dict in enumerate(read_truth(truth_file_name)):
        uniform_deviate = galsim.UniformDeviate(RANDOM_SEED + index)
        for band_index in xrange(5):
            if truth_dict['star_or_galaxy'] == 'galaxy':
                galaxy = create_galaxy_from_truth_parameters(
                    truth_dict,
                    GALAXY_RELATIVE_INTENSITIES[band_index],
                )
                final_light_source = galsim.Convolve([galaxy, make_psf(1)])
            else:
                point_source_plus_psf = make_psf(
                    float(truth_dict['reference_band_flux_nmgy'])
                        * STAR_RELATIVE_INTENSITIES[band_index] * COUNTS_PER_NMGY
                )
                final_light_source = apply_shift(point_source_plus_psf, truth_dict)

            image = galsim.ImageF(STAMP_SIZE_PX, STAMP_SIZE_PX, scale=ARCSEC_PER_PIXEL)
            final_light_source.drawImage(image)
            image.array[:] = image.array[:] + float(truth_dict['sky_level_nmgy']) * COUNTS_PER_NMGY
            if truth_dict['add_noise'] == '1':
                add_noise(image, uniform_deviate)
            images.append(image)

    if not os.path.exists('output'):
        os.mkdir('output')
    image_file_name = os.path.join('output', 'galsim_test_images.fits')
    galsim.fits.writeMulti(images, image_file_name)
    _logger.info('Wrote multi-extension FITS images to %r', image_file_name)

if __name__ == "__main__":
    main()
