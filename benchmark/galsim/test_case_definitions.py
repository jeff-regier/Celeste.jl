import collections

import galsim

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

    # `light_source` must satisfy the LightSource interface
    def add(self, light_source):
        self._light_sources.append(light_source)

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

# Test cases follow. Each is a function which accepts a test case and adds LightSources and/or sets
# parameters. Each must be decorated with @galsim_test_case. Function name will be included in the
# FITS header for reference.

@galsim_test_case
def simple_star(test_case):
    test_case.add(Star())

@galsim_test_case
def star_position_1(test_case):
    test_case.add(Star().offset_arcsec(-2, 0))

@galsim_test_case
def star_position_2(test_case):
    test_case.add(Star().offset_arcsec(0, 2))

@galsim_test_case
def dim_star(test_case):
    test_case.add(Star().reference_band_flux_nmgy(20))

@galsim_test_case
def bright_star(test_case):
    test_case.add(Star().reference_band_flux_nmgy(80))

@galsim_test_case
def different_color_star(test_case):
    test_case.add(Star().flux_relative_to_reference_band([0.2, 0.8, 1, 1.6, 1.3]))

@galsim_test_case
def star_with_noise(test_case):
    test_case.add(Star().offset_arcsec(-1, 1).reference_band_flux_nmgy(20))
    test_case.sky_level_nmgy = 0.1
    test_case.include_noise = True

@galsim_test_case
def angle_and_axis_ratio_1(test_case):
    test_case.add(Galaxy().angle_deg(15).minor_major_axis_ratio(0.2))

@galsim_test_case
def angle_and_axis_ratio_2(test_case):
    test_case.add(Galaxy().angle_deg(160).minor_major_axis_ratio(0.4))

@galsim_test_case
def round_galaxy(test_case):
    test_case.add(Galaxy().minor_major_axis_ratio(1))

@galsim_test_case
def small_galaxy(test_case):
    test_case.add(Galaxy().half_light_radius_arcsec(0.75))

@galsim_test_case
def large_galaxy(test_case):
    test_case.add(Galaxy().half_light_radius_arcsec(2.5))

@galsim_test_case
def dim_galaxy(test_case):
    test_case.add(Galaxy().reference_band_flux_nmgy(5))

@galsim_test_case
def bright_galaxy(test_case):
    test_case.add(Galaxy().reference_band_flux_nmgy(20))

@galsim_test_case
def different_color_galaxy(test_case):
    test_case.add(Galaxy().flux_relative_to_reference_band([0.6, 0.2, 1, 1.1, 2]))

@galsim_test_case
def galaxy_with_all(test_case):
    test_case.add(
        Galaxy().offset_arcsec(0.3, -0.7)
            .angle_deg(15)
            .minor_major_axis_ratio(0.4)
            .half_light_radius_arcsec(2.5)
            .reference_band_flux_nmgy(15)
    )

@galsim_test_case
def galaxy_with_noise(test_case):
    galaxy_with_all(test_case)
    test_case.include_noise = True

@galsim_test_case
def galaxy_with_low_background(test_case):
    galaxy_with_noise(test_case)
    test_case.sky_level_nmgy = 0.1

@galsim_test_case
def galaxy_with_high_background(test_case):
    galaxy_with_noise(test_case)
    test_case.sky_level_nmgy = 0.3

@galsim_test_case
def overlapping_stars(test_case):
    test_case.add(Star().offset_arcsec(-3, 0))
    test_case.add(Star().offset_arcsec(3, 0))

@galsim_test_case
def overlapping_galaxies(test_case):
    test_case.add(Galaxy().offset_arcsec(-2, -2).angle_deg(135).minor_major_axis_ratio(0.2))
    test_case.add(Galaxy().offset_arcsec(3, 3).angle_deg(35).minor_major_axis_ratio(0.5))

@galsim_test_case
def overlapping_star_and_galaxy(test_case):
    test_case.add(Star().offset_arcsec(-5, 0))
    test_case.add(Galaxy().offset_arcsec(2, 2).angle_deg(35).minor_major_axis_ratio(0.5))

@galsim_test_case
def three_sources_two_overlap(test_case):
    test_case.add(Star().offset_arcsec(-5, 5))
    test_case.add(
        Galaxy().offset_arcsec(2, 5)
            .angle_deg(35)
            .minor_major_axis_ratio(0.2)
    )
    test_case.add(Star().offset_arcsec(10, -10))

@galsim_test_case
def three_sources_all_overlap(test_case):
    overlapping_star_and_galaxy(test_case)
    test_case.add(Star().offset_arcsec(8, -1))

@galsim_test_case
def smaller_psf(test_case):
    test_case.psf_sigma_pixels = 2
    test_case.add(Star())

@galsim_test_case
def larger_psf(test_case):
    test_case.psf_sigma_pixels = 6
    test_case.add(Star())
