import csv
import logging
import math
import os

import galsim

_logger = logging.getLogger(__name__) 

TRUTH_COLUMNS = ['left', 'bottom', 'angle_radians', 'ellipticity', 'world_center_x', 'world_center_y']

# Galaxy parameters
GALAXY_RADIUS=0.85 # arcsec
GALAXY_FLUX = 1.e5 # total counts in image
GAL_ELLIP_MIN = 0.2             # Minimum value of minor/major axis ratio

# PSF parameters
ATMOS_FWHM=2.1         # arcsec
ATMOS_E = 0.13         # 
ATMOS_BETA = 0.81      # radians
OPT_DEFOCUS=0.53       # wavelengths
OPT_A1=-0.29           # wavelengths
OPT_A2=0.12            # wavelengths
OPT_C1=0.64            # wavelengths
OPT_C2=-0.33           # wavelengths
OPT_OBSCURATION=0.3    # linear scale size of secondary mirror obscuration
LAM = 800              # nm    NB: don't use lambda - that's a reserved word.
TEL_DIAM = 4.          # meters

# Other parameters
PIXEL_SCALE = 0.75  # arcsec / pixel
SKY_LEVEL = 0. # TODO

SHIFT_RADIUS = 5.0              # arcsec

NX_TILES = 1
NY_TILES = 1
STAMP_XSIZE = 48
STAMP_YSIZE = 48

RANDOM_SEED = 6

def make_galaxy():
    return galsim.Exponential(scale_radius=GALAXY_RADIUS, flux=GALAXY_FLUX)

def make_atmospheric_psf():
    atmos = galsim.Kolmogorov(fwhm=ATMOS_FWHM)
    return atmos.shear(e=ATMOS_E, beta=ATMOS_BETA*galsim.radians)

def make_optical_psf():
    # The first argument of OpticalPSF below is lambda/diam (wavelength of light / telescope
    # diameter), which needs to be in the same units used to specify the image scale.  We are using
    # arcsec for that, so we have to self-consistently use arcsec here, using the following
    # calculation:
    lam_over_diam = LAM * 1.e-9 / TEL_DIAM # radians
    lam_over_diam *= 206265  # arcsec
    _logger.debug('Calculated lambda over diam = %f arcsec', lam_over_diam)
    return galsim.OpticalPSF(
        lam_over_diam, 
        defocus=OPT_DEFOCUS,
        coma1=OPT_C1,
        coma2=OPT_C2,
        astig1=OPT_A1,
        astig2=OPT_A2,
        obscuration=OPT_OBSCURATION,
    )

def make_psf():
    return galsim.Convolve([make_atmospheric_psf(), make_optical_psf()])

def get_stamp_subimage(full_image, x_index, y_index):
    bounds = galsim.BoundsI(
        x_index * STAMP_XSIZE + 1,
        (x_index + 1) * STAMP_XSIZE - 1,
        y_index * STAMP_YSIZE + 1,
        (y_index + 1) * STAMP_YSIZE - 1,
    )
    return full_image[bounds]

def apply_shear(galaxy, uniform_deviate):
    beta = uniform_deviate() * 2. * math.pi * galsim.radians
    ellip = uniform_deviate() * (1 - GAL_ELLIP_MIN) + GAL_ELLIP_MIN
    return galaxy.shear(q=ellip, beta=beta), beta, ellip

def apply_shift(galaxy, uniform_deviate):
    rsq = 2 * SHIFT_RADIUS**2
    while (rsq > SHIFT_RADIUS**2):
        dx = (10 * uniform_deviate() - 5) * SHIFT_RADIUS
        dy = (2 * uniform_deviate() - 1) * SHIFT_RADIUS
        rsq = dx**2 + dy**2
    return galaxy.shift(dx, dy), dx, dy

def add_noise(image, uniform_deviate):
    noise = galsim.PoissonNoise(uniform_deviate, sky_level=SKY_LEVEL)
    image.addNoise(noise)

def write_truth(file_name, rows):
    with open(file_name, 'w') as truth_stream:
        writer = csv.DictWriter(truth_stream, TRUTH_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    gal = make_galaxy()
    psf = make_psf()

    gal_image = galsim.ImageF(
        STAMP_XSIZE * NX_TILES - 1,
        STAMP_YSIZE * NY_TILES - 1,
        scale=PIXEL_SCALE,
    )

    _logger.info('Generating %d galaxy images', NX_TILES * NY_TILES)
    k = 0
    truth_rows = []
    for iy in range(NY_TILES):
        for ix in range(NX_TILES):
            uniform_deviate = galsim.UniformDeviate(RANDOM_SEED + k + 1)
            k += 1

            this_galaxy, beta, ellipticity = apply_shear(gal, uniform_deviate)
            this_galaxy, shift_x, shift_y = apply_shift(this_galaxy, uniform_deviate)

            stamp_image = get_stamp_subimage(gal_image, ix, iy)
            final_gal = this_galaxy #galsim.Convolve([psf, this_galaxy])
            final_gal.drawImage(stamp_image)
            add_noise(stamp_image, uniform_deviate)

            truth_rows.append({
                'left': ix * STAMP_XSIZE,
                'bottom': iy * STAMP_YSIZE,
                'angle_radians': beta.rad(),
                'ellipticity': ellipticity,
                'world_center_x': (STAMP_XSIZE / 2 + shift_x) * PIXEL_SCALE,
                'world_center_y': (STAMP_YSIZE / 2 + shift_y) * PIXEL_SCALE,
            })

    if not os.path.exists('output'):
        os.mkdir('output')

    image_file_name = os.path.join('output', 'galsim_test_image.fits')
    gal_image.write(image_file_name)
    _logger.info('Wrote image to %r', image_file_name)

    truth_file_name = os.path.join('output', 'galsim_truth.csv')
    write_truth(truth_file_name, truth_rows)
    _logger.info('Wrote truth values to %r', truth_file_name)

if __name__ == "__main__":
    main()
