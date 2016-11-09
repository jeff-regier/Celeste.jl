import galsim

# populated by the `galsim_test_case` decorator
TEST_CASE_FNS = []

ARCSEC_PER_PIXEL = 0.75
SHIFT_RADIUS_ARCSEC = ARCSEC_PER_PIXEL
PSF_SIGMA_PIXELS = 4
STAMP_SIZE_PX = 48
COUNTS_PER_NMGY = 1000.0 # a.k.a. "iota" in Celeste

# intensity (flux) relative to third band (= "a" band = reference)
# see GalsimBenchmark.typical_band_relative_intensities()
# these are taken from the current dominant component of the lognormal prior on c_s for stars
STAR_RELATIVE_INTENSITIES = [
    0.1330,
    0.5308,
    1,
    1.3179,
    1.5417,
]
# these are taken from the current dominant component of the lognormal prior on c_s for galaxies
GALAXY_RELATIVE_INTENSITIES = [
    0.4013,
    0.4990,
    1,
    1.4031,
    1.7750,
]

def psf_sigma_arcsec(): return PSF_SIGMA_PIXELS * ARCSEC_PER_PIXEL

# fields and logic shared between stars and galaxies
class CommonFields(object):
    def __init__(self):
        self.offset_from_center_world_coords = (0, 0)
        self.reference_band_flux_nmgy = 40

    def position_world_coords(self):
        image_center_in_world_coordinates = (STAMP_SIZE_PX + 1) / 2.0 * ARCSEC_PER_PIXEL
        return (
            image_center_in_world_coordinates + self.offset_from_center_world_coords[0],
            image_center_in_world_coordinates + self.offset_from_center_world_coords[1],
        )

    def add_header_fields(self, header, star_or_galaxy):
        position = self.position_world_coords()
        header['CL_CENTX'] = (position[0], 'X center in world coordinates')
        header['CL_CENTY'] = (position[1], 'Y center in world coordinates')
        header['CL_FLUX'] = (
            self.reference_band_flux_nmgy,
            'reference (=3) band brightness (nMgy)',
        )
        header['CL_STGAL'] = (star_or_galaxy, '"star" or "galaxy"?')

# this essentially just serves a documentation purpose
class LightSource(object):
    def get_galsim_light_source(self, band_index):
        raise NotImplementedError

    def add_header_fields(self, header):
        raise NotImplementedError

class Star(LightSource):
    def __init__(self):
        self._common_fields = CommonFields()

    def offset_world_coords(self, x, y):
        self._common_fields.offset_from_center_world_coords = (x, y)
        return self

    def reference_band_flux_nmgy(self, flux):
        self._common_fields.reference_band_flux_nmgy = flux
        return self

    def get_galsim_light_source(self, band_index):
        flux_counts = (
            self._common_fields.reference_band_flux_nmgy * STAR_RELATIVE_INTENSITIES[band_index]
                * COUNTS_PER_NMGY
        )
        offset = self._common_fields.offset_from_center_world_coords
        return (
            galsim.Gaussian(flux=flux_counts, sigma=psf_sigma_arcsec())
                .shift(offset[0], offset[1])
        )

    def add_header_fields(self, header):
        self._common_fields.add_header_fields(header, 'star')

class Galaxy(LightSource):
    def __init__(self):
        self._common_fields = CommonFields()
        self._angle_deg = 0
        self._minor_major_axis_ratio = 0.4
        self._half_light_radius_arcsec = 6

    def offset_world_coords(self, x, y):
        self._common_fields.offset_from_center_world_coords = (x, y)
        return self

    def reference_band_flux_nmgy(self, flux):
        self._common_fields.reference_band_flux_nmgy = flux
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

    def get_galsim_light_source(self, band_index):
        flux_counts = (
            self._common_fields.reference_band_flux_nmgy * GALAXY_RELATIVE_INTENSITIES[band_index]
                * COUNTS_PER_NMGY
        )
        galaxy = (
            galsim.Exponential(half_light_radius=self._half_light_radius_arcsec, flux=flux_counts)
                .shear(q=self._minor_major_axis_ratio, beta=self._angle_deg * galsim.degrees)
                .shift(self._common_fields.offset_from_center_world_coords)
        )
        psf = galsim.Gaussian(flux=1, sigma=psf_sigma_arcsec())
        return galsim.Convolve([galaxy, psf])

    def add_header_fields(self, header):
        self._common_fields.add_header_fields(header, 'galaxy')
        header['CL_ANGLE'] = (self._angle_deg, 'major axis angle (degrees from x-axis)')
        header['CL_RATIO'] = (self._minor_major_axis_ratio, 'minor/major axis ratio')
        header['CL_HLRAD'] = (self._half_light_radius_arcsec, 'half-light radius (arcsec)')

# A complete description of a GalSim test image, along with logic to generate the image and the
# "ground truth" header fields
class GalSimTestCase(object):
    def __init__(self):
        self._light_sources = []
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
        image = galsim.ImageF(STAMP_SIZE_PX, STAMP_SIZE_PX, scale=ARCSEC_PER_PIXEL)
        for light_source in self._light_sources:
            light_source.get_galsim_light_source(band_index).drawImage(image)
        self._add_sky_background(image)
        self._add_noise(image, uniform_deviate)
        return image

    def get_fits_header(self, case_index, band_index):
        header = {
            'CL_SKY': (self.sky_level_nmgy, '"epsilon" sky level (nMgy each px)'),
            'CL_NOISE': (self.include_noise, 'was Poisson noise added?'),
            'CL_DESCR': (self.comment, 'comment'),
            'CL_CASEI': (case_index + 1, 'test case index'),
            'CL_BAND': (band_index + 1, 'color band'),
            'CL_IOTA': (COUNTS_PER_NMGY, 'counts per nMgy'),
        }
        for light_source in self._light_sources:
            light_source.add_header_fields(header)
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
    test_case.add(Star().offset_world_coords(-2, 0))

@galsim_test_case
def star_position_2(test_case):
    test_case.add(Star().offset_world_coords(0, 2))

@galsim_test_case
def dim_star(test_case):
    test_case.add(Star().reference_band_flux_nmgy(20))

@galsim_test_case
def bright_star(test_case):
    test_case.add(Star().reference_band_flux_nmgy(80))

@galsim_test_case
def star_with_noise(test_case):
    test_case.add(Star().offset_world_coords(-1, 1).reference_band_flux_nmgy(20))
    test_case.sky_level_nmgy = 0.1
    test_case.include_noise = True

@galsim_test_case
def angle_and_axis_ratio_1(test_case):
    test_case.add(Galaxy().angle_deg(15).minor_major_axis_ratio(0.2))

@galsim_test_case
def angle_and_axis_ratio_2(test_case):
    test_case.add(Galaxy().angle_deg(160).minor_major_axis_ratio(0.4))

@galsim_test_case
def small_galaxy(test_case):
    test_case.add(Galaxy().half_light_radius_arcsec(3))

@galsim_test_case
def large_galaxy(test_case):
    test_case.add(Galaxy().half_light_radius_arcsec(10))

@galsim_test_case
def dim_galaxy(test_case):
    test_case.add(Galaxy().reference_band_flux_nmgy(5))

@galsim_test_case
def bright_galaxy(test_case):
    test_case.add(Galaxy().reference_band_flux_nmgy(20))

@galsim_test_case
def galaxy_with_all(test_case):
    test_case.add(
        Galaxy().offset_world_coords(0.3, -0.7)
            .angle_deg(15)
            .minor_major_axis_ratio(0.4)
            .half_light_radius_arcsec(9)
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
def galaxy_with_low_background(test_case):
    galaxy_with_noise(test_case)
    test_case.sky_level_nmgy = 0.3
