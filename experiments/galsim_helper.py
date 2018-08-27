import collections
import hashlib
import logging
import os
import sys

import astropy.io.fits
import galsim

_logger = logging.getLogger(__name__)

RANDOM_SEED = 1234
FITS_COMMENT_PREPEND = 'Celeste: '

ARCSEC_PER_DEGREE = 3600.

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

class WorldCoordinate(object):
    def __init__(self, ra, dec):
        self.ra = ra
        self.dec = dec

    def as_galsim_position(self):
        return galsim.PositionD(
            self.ra,
            self.dec,
        )

    def add(self, right_ascension_offset_deg, declination_offset_deg):
        return WorldCoordinate(
            self.ra + right_ascension_offset_deg,
            self.dec + declination_offset_deg,
        )

class ImageParameters(object):
    def __init__(self):
        # "width" corresponds to declination, "height" to right ascension
        self.width_px = 96
        self.height_px = 96
        # 0.396 = resolution of SDSS images (https://github.com/jeff-regier/Celeste.jl/pull/411)
        self.arcsec_per_pixel = 0.396
        self.world_origin = WorldCoordinate(0, 0)
        self.band_nelec_per_nmgy = [1000.0 for _ in range(5)]

    def degrees_per_pixel(self):
        return self.arcsec_per_pixel / ARCSEC_PER_DEGREE

    def get_image_center_world_coordinates(self):
        width_deg = self.width_px * self.degrees_per_pixel()
        height_deg = self.height_px * self.degrees_per_pixel()
        return self.world_origin.add(height_deg / 2., width_deg / 2.)

class AbsolutePosition(object):
    def __init__(self, ra, dec):
        self._position = WorldCoordinate(ra, dec)

    def get_position(self, image_parameters):
        return self._position

class OffsetFromCenterPosition(object):
    def __init__(self, right_ascension_offset_deg, declination_offset_deg):
        self._offset = WorldCoordinate(right_ascension_offset_deg, declination_offset_deg)

    def get_position(self, image_parameters):
        return image_parameters.get_image_center_world_coordinates().add(
            self._offset.ra,
            self._offset.dec,
        )

# fields and logic shared between stars and galaxies
class CommonFields(object):
    def __init__(self):
        self.position = OffsetFromCenterPosition(0, 0) # or an AbsolutePosition
        self.flux_r_nmgy = 10
        # relative flux in each band defines "color" of light sources
        self._flux_relative_to_reference_band = DEFAULT_GALAXY_RELATIVE_INTENSITIES

    def set_offset_from_center_arcsec(self, right_ascension_offset_arcsec, declination_offset_arcsec):
        self.position = OffsetFromCenterPosition(
            right_ascension_offset_arcsec / ARCSEC_PER_DEGREE,
            declination_offset_arcsec / ARCSEC_PER_DEGREE,
        )

    def set_world_coordinates_deg(self, ra, dec):
        self.position = AbsolutePosition(ra, dec)

    def get_world_offset(self, image_parameters):
        """Get offset from image center, in world coordinates (degrees)

        We do this because GalSim images are located at the center of the image by default, and the
        only way I've found to change the position of an object is using `shift()` (I don't know how
        to set an absolute position in world coords).
        """
        image_center = image_parameters.get_image_center_world_coordinates()
        world_position = self.position.get_position(image_parameters)
        return world_position.add(-image_center.ra, -image_center.dec)

    def set_flux_relative_to_reference_band(self, relative_flux):
        assert len(relative_flux) == 5
        assert relative_flux[2] == 1
        self._flux_relative_to_reference_band = relative_flux

    def get_flux_nmgy(self, band_index):
        return (
            self.flux_r_nmgy * self._flux_relative_to_reference_band[band_index]
        )

    def add_header_fields(self, header, index_str, image_parameters, star_or_galaxy):
        position = self.position.get_position(image_parameters)
        header['CLRA' + index_str] = (position.ra, 'Center right ascension, deg')
        header['CLDEC' + index_str] = (position.dec, 'Center declination, deg')
        header['CLFLX' + index_str] = (
            self.flux_r_nmgy,
            'reference (=3) band brightness (nMgy)',
        )
        header['CLC12' + index_str] = (
            self._flux_relative_to_reference_band[1] / self._flux_relative_to_reference_band[0],
            'ratio of flux in band 2 to band 1',
        )
        header['CLC23' + index_str] = (
            1 / self._flux_relative_to_reference_band[1],
            'ratio of flux in band 3 to band 2',
        )
        header['CLC34' + index_str] = (
            self._flux_relative_to_reference_band[3],
            'ratio of flux in band 4 to band 3',
        )
        header['CLC45' + index_str] = (
            self._flux_relative_to_reference_band[4] / self._flux_relative_to_reference_band[3],
            'ratio of flux in band 5 to band 4',
        )
        header['CLTYP' + index_str] = (star_or_galaxy, '"star" or "galaxy"?')

# this essentially just serves a documentation purpose
class LightSource(object):
    def get_galsim_light_source(self, band_index):
        raise NotImplementedError

    def add_header_fields(self, header, index_str, image_parameters):
        raise NotImplementedError

class Star(LightSource):
    def __init__(self):
        self._common_fields = CommonFields()
        self._common_fields.set_flux_relative_to_reference_band(DEFAULT_STAR_RELATIVE_INTENSITIES)

    def offset_arcsec(self, x, y):
        self._common_fields.set_offset_from_center_arcsec(x, y)
        return self

    def world_coordinates_deg(self, ra, dec):
        self._common_fields.set_world_coordinates_deg(ra, dec)
        return self

    def flux_r_nmgy(self, flux):
        self._common_fields.flux_r_nmgy = flux
        return self

    def flux_relative_to_reference_band(self, relative_flux):
        self._common_fields.set_flux_relative_to_reference_band(relative_flux)
        return self

    def get_galsim_light_source(self, band_index, psf_sigma_degrees, image_parameters):
        flux_nelec = (
            self._common_fields.get_flux_nmgy(band_index)
            * image_parameters.band_nelec_per_nmgy[band_index]
        )
        return (
            galsim.Gaussian(flux=flux_nelec, sigma=psf_sigma_degrees)
                .shift(self._common_fields.get_world_offset(image_parameters).as_galsim_position())
        )

    def add_header_fields(self, header, index_str, image_parameters):
        self._common_fields.add_header_fields(header, index_str, image_parameters, 'star')

class Galaxy(LightSource):
    def __init__(self):
        self._common_fields = CommonFields()
        self._common_fields.flux_r_nmgy = 10
        self._gal_angle_deg = 0
        self._axis_ratio = 0.4
        self._half_light_radius_arcsec = 1.5
        self._gal_frac_dev = 0.0

    def offset_arcsec(self, x, y):
        self._common_fields.set_offset_from_center_arcsec(x, y)
        return self

    def world_coordinates_deg(self, ra, dec):
        self._common_fields.set_world_coordinates_deg(ra, dec)
        return self

    def flux_r_nmgy(self, flux):
        self._common_fields.flux_r_nmgy = flux
        return self

    def flux_relative_to_reference_band(self, relative_flux):
        self._common_fields.set_flux_relative_to_reference_band(relative_flux)
        return self

    def gal_angle_deg(self, angle):
        self._gal_angle_deg = angle
        return self

    def axis_ratio(self, ratio):
        self._axis_ratio = ratio
        return self

    def half_light_radius_arcsec(self, radius):
        self._half_light_radius_arcsec = radius
        return self

    def gal_frac_dev(self, weight):
        self._gal_frac_dev = weight
        return self

    def get_galsim_light_source(self, band_index, psf_sigma_degrees, image_parameters):
        def apply_shear_and_shift(galaxy):
            return (
                galaxy.shear(
                    q=self._axis_ratio,
                    beta=(90. - self._gal_angle_deg) * galsim.degrees,
                )
                .shift(self._common_fields.get_world_offset(image_parameters).as_galsim_position())
            )

        flux_nelec = (
            self._common_fields.get_flux_nmgy(band_index)
            * image_parameters.band_nelec_per_nmgy[band_index]
        )
        half_light_radius_deg = self._half_light_radius_arcsec / ARCSEC_PER_DEGREE
        exponential_profile = apply_shear_and_shift(
            galsim.Exponential(
                half_light_radius=half_light_radius_deg,
                flux=flux_nelec * (1 - self._gal_frac_dev),
            )
        )
        de_vaucouleurs_profile = apply_shear_and_shift(
            galsim.DeVaucouleurs(
                half_light_radius=half_light_radius_deg,
                flux=flux_nelec * self._gal_frac_dev,
            )
        )
        galaxy = exponential_profile + de_vaucouleurs_profile
        psf = galsim.Gaussian(flux=1, sigma=psf_sigma_degrees)
        return galsim.Convolve([galaxy, psf])

    def add_header_fields(self, header, index_str, image_parameters):
        self._common_fields.add_header_fields(header, index_str, image_parameters, 'galaxy')
        header['CLANG' + index_str] = (self._gal_angle_deg, 'maj axis angle (deg from +dec -> +ra)')
        header['CLRTO' + index_str] = (self._axis_ratio, 'minor/major axis ratio')
        header['CLRDA' + index_str] = (
            self._half_light_radius_arcsec,
            'half-light radius (arcsec)',
        )
        header['CLRDP' + index_str] = (
            self._half_light_radius_arcsec / image_parameters.arcsec_per_pixel,
            'half-light radius (pixels)',
        )
        header['CLDEV' + index_str] = (
            self._gal_frac_dev,
            'de Vaucouleurs mixture weight',
        )

# A complete description of a GalSim test image, along with logic to generate the image and the
# "ground truth" header fields
class GalSimTestCase(object):
    def __init__(self):
        self._light_sources = []
        self.image_parameters = ImageParameters()
        self.psf_sigma_pixels = 4
        self.band_sky_level_nmgy = [0.01 for _ in range(5)] # very low noise by default
        self.include_noise = False
        self.comment = None

    def set_dimensions(self, width_px, height_px):
        self.image_parameters.width_px = width_px
        self.image_parameters.height_px = height_px

    def set_resolution(self, arcsec_per_pixel):
        self.image_parameters.arcsec_per_pixel = arcsec_per_pixel

    def set_world_origin(self, ra, dec):
        self.image_parameters.world_origin = WorldCoordinate(ra, dec)

    def set_band_nelec_per_nmgy(self, band_nelec_per_nmgy):
        self.image_parameters.band_nelec_per_nmgy = band_nelec_per_nmgy

    def get_resolution(self):
        return self.image_parameters.arcsec_per_pixel

    def add_star(self):
        star = Star()
        self._light_sources.append(star)
        return star

    def add_galaxy(self):
        galaxy = Galaxy()
        self._light_sources.append(galaxy)
        return galaxy

    def _add_sky_background(self, image, band_index, uniform_deviate):
        sky_level_nelec = (
            self.band_sky_level_nmgy[band_index]
            * self.image_parameters.band_nelec_per_nmgy[band_index]
        )
        if self.include_noise:
            poisson_deviate = galsim.PoissonDeviate(uniform_deviate, mean=sky_level_nelec)
            image.addNoise(galsim.DeviateNoise(poisson_deviate))
        else:
            image.array[:] = image.array + sky_level_nelec

    def construct_image(self, band_index, uniform_deviate):
        world_origin = self.image_parameters.world_origin.as_galsim_position()
        degrees_per_pixel = self.image_parameters.degrees_per_pixel()
        wcs = (
            # Here we implement the confusing mapping X <-> Dec, Y <-> RA
            galsim.JacobianWCS(0, degrees_per_pixel, degrees_per_pixel, 0)
                .withOrigin(galsim.PositionI(0, 0), world_origin=world_origin)
        )
        image = galsim.ImageF(
            self.image_parameters.width_px,
            self.image_parameters.height_px,
            wcs=wcs,
        )
        for index, light_source in enumerate(self._light_sources):
            sys.stdout.write('Band {} source {}\r'.format(band_index + 1, index + 1))
            sys.stdout.flush()
            galsim_light_source = light_source.get_galsim_light_source(
                band_index,
                self.psf_sigma_pixels * self.image_parameters.degrees_per_pixel(),
                self.image_parameters,
            )
            galsim_light_source.drawImage(
                image,
                add_to_image=True,
                method='phot',
                max_extra_noise=
                    self.band_sky_level_nmgy[band_index]
                    * self.image_parameters.band_nelec_per_nmgy[band_index] / 1000.0,
                rng=uniform_deviate,
            )
        self._add_sky_background(image, band_index, uniform_deviate)
        return image

    def get_fits_header(self, case_index, band_index):
        # FITS header fields will be too long if there's a four-digit index
        assert len(self._light_sources) < 1000
        header = collections.OrderedDict([
            ('CLCASEI', (case_index + 1, 'test case index')),
            ('CLDESCR', (self.comment, 'comment')),
            ('CLIOTA', (self.image_parameters.band_nelec_per_nmgy[band_index], 'nelec per nMgy')),
            ('CLSKY', (self.band_sky_level_nmgy[band_index], 'background level (nMgy each px)')),
            ('CLNOISE', (self.include_noise, 'was Poisson noise added?')),
            ('CLSIGMA', (self.psf_sigma_pixels, 'Gaussian PSF sigma (px)')),
            ('CLBAND', (band_index + 1, 'color band')),
            ('CLNSRC', (len(self._light_sources), 'number of sources')),
            ('CLRES', (self.image_parameters.degrees_per_pixel(), 'resolution (degrees/px)')),
        ])
        for source_index, light_source in enumerate(self._light_sources):
            index_str = '{:03d}'.format(source_index + 1)
            light_source.add_header_fields(header, index_str, self.image_parameters)
        return header

def add_header_to_hdu(hdu, header_dict):
    header = galsim.fits.FitsHeader(hdu.header)
    for name, (value, comment) in header_dict.items():
        header[name] = (value, FITS_COMMENT_PREPEND + comment)

def ensure_containing_directory_exists(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))

def save_multi_extension_fits(hdu_list, filename):
    ensure_containing_directory_exists(filename)
    if os.path.exists(filename):
        os.remove(filename)
    hdu_list.writeto(filename)

def write_latest_filename(output_label, latest_filename):
    latest_fits_filename_holder = os.path.join(
        'latest_filenames', 'latest_{}.txt'.format(output_label)
    )
    ensure_containing_directory_exists(latest_fits_filename_holder)
    with open(latest_fits_filename_holder, 'w') as stream:
        stream.write(latest_filename)
        stream.write('\n')
    _logger.info('Updated %r', latest_fits_filename_holder)

def generate_fits_file(output_label, test_case_callbacks):
    _logger.info('Generating %d test cases', len(test_case_callbacks))
    fits_hdus = astropy.io.fits.HDUList()
    for case_index, test_case_fn in enumerate(test_case_callbacks):
        _logger.info('  Generating case %s', test_case_fn.__name__)
        uniform_deviate = galsim.UniformDeviate(RANDOM_SEED + case_index)
        test_case = GalSimTestCase()
        test_case_fn(test_case)
        for band_index in range(5):
            image = test_case.construct_image(band_index, uniform_deviate)
            galsim.fits.write(image, hdu_list=fits_hdus)
            fits_hdus[-1].name = '{}_{}'.format(test_case_fn.__name__, band_index + 1)
            add_header_to_hdu(fits_hdus[-1], test_case.get_fits_header(case_index, band_index))

    image_file_name = os.path.join(os.getcwd(), output_label + '.fits')
    save_multi_extension_fits(fits_hdus, image_file_name)
    print('Wrote multi-extension FITS file to', image_file_name)
