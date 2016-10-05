import csv
import logging
import math
import os

import galsim

_logger = logging.getLogger(__name__) 

TRUTH_COLUMNS = ['left', 'bottom', 'angle_radians', 'ellipticity', 'world_center_x', 'world_center_y']

PIXEL_SCALE = 0.75 # arcsec / pixel
SHIFT_RADIUS_ARCSEC = PIXEL_SCALE
PSF_SIGMA_ARCSEC = PIXEL_SCALE # so that PSF sigma = 1 pixel
STAMP_SIZE_PX = 48

# intensity (flux) relative to third band (= "a" band = reference)
# these are taken from the current second component of the lognormal prior on c_s for galaxies
BAND_RELATIVE_INTENSITIES = [
    0.219,
    0.465,
    1,
    1.421,
    1.846,
]

RANDOM_SEED = 1234

def make_galaxy(half_light_radius_arcsec, flux_counts):
    return galsim.Exponential(half_light_radius=half_light_radius_arcsec, flux=flux_counts)

def make_psf():
    return galsim.Gaussian(flux=1., sigma=PSF_SIGMA_ARCSEC)

def apply_shear(galaxy, angle_degrees, minor_major_axis_ratio):
    return galaxy.shear(q=minor_major_axis_ratio, beta=angle_degrees * galsim.degrees)

def apply_shift(galaxy, shift_x_world_coords, shift_y_world_coords):
    return galaxy.shift(shift_x_world_coords, shift_y_world_coords)

def add_noise(image, sky_level, uniform_deviate):
    noise = galsim.PoissonNoise(uniform_deviate, sky_level=sky_level)
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
        psf = make_psf()
        for band_index in xrange(5):
            image = galsim.ImageF(STAMP_SIZE_PX, STAMP_SIZE_PX, scale=PIXEL_SCALE)
            galaxy = make_galaxy(
                float(truth_dict['half_light_radius_arcsec']),
                float(truth_dict['flux_counts']) * BAND_RELATIVE_INTENSITIES[band_index],
            )
            galaxy = apply_shear(
                galaxy,
                float(truth_dict['angle_degrees']),
                float(truth_dict['minor_major_axis_ratio']),
            )
            galaxy = apply_shift(
                galaxy,
                float(truth_dict['world_center_x']) - STAMP_SIZE_PX * PIXEL_SCALE / 2.0,
                float(truth_dict['world_center_y']) - STAMP_SIZE_PX * PIXEL_SCALE / 2.0,
            )
            final_light_source = galsim.Convolve([galaxy, psf])
            final_light_source.drawImage(image)
            if truth_dict['add_noise'] == '1':
                add_noise(image, float(truth_dict['sky_level']), uniform_deviate)
            images.append(image)

        if not os.path.exists('output'):
            os.mkdir('output')

    image_file_name = os.path.join('output', 'galsim_test_images.fits')
    galsim.fits.writeMulti(images, image_file_name)
    _logger.info('Wrote multi-extension FITS images to %r', image_file_name)

if __name__ == "__main__":
    main()
