import collections
import hashlib
import logging
import os

import astropy.io.fits
import galsim

_logger = logging.getLogger(__name__) 

RANDOM_SEED = 1234
FITS_COMMENT_PREPEND = 'Celeste: '

# populated by the `galsim_test_case` decorator
TEST_CASE_FNS = []

ARCSEC_PER_DEGREE = 3600.
ARCSEC_PER_PIXEL = 0.396 # the value used in SDSS (https://github.com/jeff-regier/Celeste.jl/pull/411)
DEGREES_PER_PIXEL = ARCSEC_PER_PIXEL / ARCSEC_PER_DEGREE
STAMP_SIZE_PX = 96
COUNTS_PER_NMGY = 1000.0 # a.k.a. "iota" in Celeste

# intensity (flux) relative to third band (= "a" band = reference)
# see GalsimBenchmark.typical_band_relative_intensities()
# these are taken from the current dominant component of the lognormal prior on c_s for stars
DEFAULT_STAR_RELATIVE_INTENSITIES = [
    0.1330,
    0.5308,
    1,
    1.3179,
    1.5417,
]
# these are taken from the current dominant component of the lognormal prior on c_s for galaxies
DEFAULT_GALAXY_RELATIVE_INTENSITIES = [
    0.4013,
    0.4990,
    1,
    1.4031,
    1.7750,
]

# fields and logic shared between stars and galaxies
class CommonFields(object):
    def __init__(self):
        self.offset_from_center_arcsec = (0, 0)
        self.reference_band_flux_nmgy = 40
        # relative flux in each band defines "color" of light sources
        self.flux_relative_to_reference_band = DEFAULT_GALAXY_RELATIVE_INTENSITIES

    def set_flux_relative_to_reference_band(self, relative_flux):
        assert len(relative_flux) == 5
        assert relative_flux[2] == 1
        self.flux_relative_to_reference_band = relative_flux

    def offset_from_center_degrees(self):
        return (
            self.offset_from_center_arcsec[0] / ARCSEC_PER_DEGREE,
            self.offset_from_center_arcsec[1] / ARCSEC_PER_DEGREE,
        )

    def position_deg(self):
        image_center_deg = (
            (STAMP_SIZE_PX + 1) / 2.0 * DEGREES_PER_PIXEL
        )
        return (
            image_center_deg + self.offset_from_center_arcsec[0] / ARCSEC_PER_DEGREE,
            image_center_deg + self.offset_from_center_arcsec[1] / ARCSEC_PER_DEGREE,
        )

    def flux_counts(self, band_index):
        return (
            self.reference_band_flux_nmgy * self.flux_relative_to_reference_band[band_index]
            * COUNTS_PER_NMGY
        )

    def add_header_fields(self, header, index_str, star_or_galaxy):
        position_deg = self.position_deg()
        header['CL_X' + index_str] = (position_deg[0], 'X center in world coordinates (deg)')
        header['CL_Y' + index_str] = (position_deg[1], 'Y center in world coordinates (deg)')
        header['CL_FLUX' + index_str] = (
            self.reference_band_flux_nmgy,
            'reference (=3) band brightness (nMgy)',
        )
        header['CL_C12_' + index_str] = (
            self.flux_relative_to_reference_band[1] / self.flux_relative_to_reference_band[0],
            'ratio of flux in band 2 to band 1',
        )
        header['CL_C23_' + index_str] = (
            1 / self.flux_relative_to_reference_band[1],
            'ratio of flux in band 3 to band 2',
        )
        header['CL_C34_' + index_str] = (
            self.flux_relative_to_reference_band[3],
            'ratio of flux in band 4 to band 3',
        )
        header['CL_C45_' + index_str] = (
            self.flux_relative_to_reference_band[4] / self.flux_relative_to_reference_band[3],
            'ratio of flux in band 5 to band 4',
        )
        header['CL_TYPE' + index_str] = (star_or_galaxy, '"star" or "galaxy"?')

# this essentially just serves a documentation purpose
class LightSource(object):
    def get_galsim_light_source(self, band_index):
        raise NotImplementedError

    def add_header_fields(self, header):
        raise NotImplementedError

class Star(LightSource):
    def __init__(self):
        self._common_fields = CommonFields()
        self._common_fields.flux_relative_to_reference_band = DEFAULT_STAR_RELATIVE_INTENSITIES

    def offset_arcsec(self, x, y):
        self._common_fields.offset_from_center_arcsec = (x, y)
        return self

    def reference_band_flux_nmgy(self, flux):
        self._common_fields.reference_band_flux_nmgy = flux
        return self

    def flux_relative_to_reference_band(self, relative_flux):
        self._common_fields.set_flux_relative_to_reference_band(relative_flux)
        return self

    def get_galsim_light_source(self, band_index, psf_sigma_degrees):
        flux_counts = self._common_fields.flux_counts(band_index)
        return (
            galsim.Gaussian(flux=flux_counts, sigma=psf_sigma_degrees)
                .shift(self._common_fields.offset_from_center_degrees())
        )

    def add_header_fields(self, header, index_str):
        self._common_fields.add_header_fields(header, index_str, 'star')

class Galaxy(LightSource):
    def __init__(self):
        self._common_fields = CommonFields()
        self._common_fields.reference_band_flux_nmgy = 10
        self._angle_deg = 0
        self._minor_major_axis_ratio = 0.4
        self._half_light_radius_arcsec = 1.5

    def offset_arcsec(self, x, y):
        self._common_fields.offset_from_center_arcsec = (x, y)
        return self

    def reference_band_flux_nmgy(self, flux):
        self._common_fields.reference_band_flux_nmgy = flux
        return self

    def flux_relative_to_reference_band(self, relative_flux):
        self._common_fields.set_flux_relative_to_reference_band(relative_flux)
        return self

    def angle_deg(self, angle):
        self._angle_deg = angle
        return self

    def minor_major_axis_ratio(self, ratio):
        self._minor_major_axis_ratio = ratio
        return self

    def half_light_radius_arcsec(self, radius):
        self._half_light_radius_arcsec = radius
        return self

    def get_galsim_light_source(self, band_index, psf_sigma_degrees):
        flux_counts = self._common_fields.flux_counts(band_index)
        half_light_radius_deg = self._half_light_radius_arcsec / ARCSEC_PER_DEGREE
        galaxy = (
            galsim.Exponential(half_light_radius=half_light_radius_deg, flux=flux_counts)
                .shear(q=self._minor_major_axis_ratio, beta=self._angle_deg * galsim.degrees)
                .shift(self._common_fields.offset_from_center_degrees())
        )
        psf = galsim.Gaussian(flux=1, sigma=psf_sigma_degrees)
        return galsim.Convolve([galaxy, psf])

    def add_header_fields(self, header, index_str):
        self._common_fields.add_header_fields(header, index_str, 'galaxy')
        header['CL_ANGL' + index_str] = (self._angle_deg, 'major axis angle (degrees from x-axis)')
        header['CL_RTIO' + index_str] = (self._minor_major_axis_ratio, 'minor/major axis ratio')
        header['CL_RADA' + index_str] = (
            self._half_light_radius_arcsec,
            'half-light radius (arcsec)',
        )
        header['CL_RADP' + index_str] = (
            self._half_light_radius_arcsec / ARCSEC_PER_PIXEL,
            'half-light radius (pixels)',
        )

# A complete description of a GalSim test image, along with logic to generate the image and the
# "ground truth" header fields
class GalSimTestCase(object):
    def __init__(self):
        self._light_sources = []
        self.psf_sigma_pixels = 4
        self.sky_level_nmgy = 0.01
        self.include_noise = False
        self.comment = None

    def add_star(self):
        star = Star()
        self._light_sources.append(star)
        return star

    def add_galaxy(self):
        galaxy = Galaxy()
        self._light_sources.append(galaxy)
        return galaxy

    def get_galsim_light_source(self, band_index):
        return sum(
            light_source.get_galsim_light_source(band_index)
            for light_source in self._light_sources
        )

    def _add_sky_background(self, image):
        image.array[:] = image.array + self.sky_level_nmgy * COUNTS_PER_NMGY

    def _add_noise(self, image, uniform_deviate):
        if self.include_noise:
            noise = galsim.PoissonNoise(uniform_deviate)
            image.addNoise(noise)

    def construct_image(self, band_index, uniform_deviate):
        image = galsim.ImageF(STAMP_SIZE_PX, STAMP_SIZE_PX, scale=DEGREES_PER_PIXEL)
        for light_source in self._light_sources:
            galsim_light_source = light_source.get_galsim_light_source(
                band_index,
                self.psf_sigma_pixels * DEGREES_PER_PIXEL,
            )
            galsim_light_source.drawImage(image, add_to_image=True)
        self._add_sky_background(image)
        self._add_noise(image, uniform_deviate)
        return image

    def get_fits_header(self, case_index, band_index):
        # FITS header fields will be too long if there's a two-digit index
        assert len(self._light_sources) < 10
        header = collections.OrderedDict([
            ('CL_CASEI', (case_index + 1, 'test case index')),
            ('CL_DESCR', (self.comment, 'comment')),
            ('CL_IOTA', (COUNTS_PER_NMGY, 'counts per nMgy')),
            ('CL_SKY', (self.sky_level_nmgy, '"epsilon" sky level (nMgy each px)')),
            ('CL_NOISE', (self.include_noise, 'was Poisson noise added?')),
            ('CL_SIGMA', (self.psf_sigma_pixels, 'Gaussian PSF sigma (px)')),
            ('CL_BAND', (band_index + 1, 'color band')),
            ('CL_NSRC', (len(self._light_sources), 'number of sources')),
        ])
        for source_index, light_source in enumerate(self._light_sources):
            light_source.add_header_fields(header, str(source_index + 1))
        return header

# just a trick to set the test case function name as the `GalSimTestCase.comment` field (for
# inclusion in the FITS header)
def galsim_test_case(fn):
    def decorated(test_case):
        fn(test_case)
        test_case.comment = fn.__name__
    decorated.__name__ = fn.__name__
    TEST_CASE_FNS.append(decorated)
    return decorated

def add_header_to_hdu(hdu, header_dict):
    header = galsim.fits.FitsHeader(hdu.header)
    for name, (value, comment) in header_dict.iteritems():
        header[name] = (value, FITS_COMMENT_PREPEND + comment)

def ensure_containing_directory_exists(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))

def save_multi_extension_fits(hdu_list, filename):
    ensure_containing_directory_exists(filename)
    if os.path.exists(filename):
        os.remove(filename)
    hdu_list.writeto(filename)

def append_md5sum_to_filename(filename):
    md5 = hashlib.md5()
    with open(filename, 'rb') as stream:
        md5.update(stream.read())
    md5sum = md5.hexdigest()
    root, extension = os.path.splitext(filename)
    new_filename = '{}_{}{}'.format(root, md5.hexdigest()[:10], extension)
    os.rename(filename, new_filename)
    return new_filename

def write_latest_filename(output_label, latest_filename):
    latest_fits_filename_holder = os.path.join(
        'latest_filenames', 'latest_{}.txt'.format(output_label)
    )
    ensure_containing_directory_exists(latest_fits_filename_holder)
    with open(latest_fits_filename_holder, 'w') as stream:
        stream.write(latest_filename)
        stream.write('\n')
    _logger.info('Updated %r', latest_fits_filename_holder)

def generate_fits_file(output_label):
    _logger.info('Generating %d test cases', len(TEST_CASE_FNS))
    fits_hdus = astropy.io.fits.HDUList()
    for case_index, test_case_fn in enumerate(TEST_CASE_FNS):
        _logger.info('  Generating case %s', test_case_fn.__name__)
        uniform_deviate = galsim.UniformDeviate(RANDOM_SEED + case_index)
        test_case = GalSimTestCase()
        test_case_fn(test_case)
        for band_index in xrange(5):
            image = test_case.construct_image(band_index, uniform_deviate)
            galsim.fits.write(image, hdu_list=fits_hdus)
            add_header_to_hdu(fits_hdus[-1], test_case.get_fits_header(case_index, band_index))

    image_file_name = os.path.join('output', output_label + '.fits')
    save_multi_extension_fits(fits_hdus, image_file_name)
    final_filename = append_md5sum_to_filename(image_file_name)
    _logger.info('Wrote multi-extension FITS file to %r', final_filename)
    write_latest_filename(output_label, os.path.basename(final_filename))
