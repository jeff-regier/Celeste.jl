import logging
import os

import astropy.io.fits
import galsim

import test_case_definitions

_logger = logging.getLogger(__name__) 

RANDOM_SEED = 1234

FITS_COMMENT_PREPEND = 'Celeste: '

def add_header_to_hdu(hdu, header_dict):
    header = galsim.fits.FitsHeader(hdu.header)
    for name, (value, comment) in header_dict.iteritems():
        header[name] = (value, FITS_COMMENT_PREPEND + comment)

def save_multi_extension_fits(hdu_list, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))
    if os.path.exists(filename):
        os.remove(filename)
    hdu_list.writeto(filename)
    _logger.info('Wrote multi-extension FITS image to %r', filename)

def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    _logger.info('Generating %d test cases', len(test_case_definitions.TEST_CASE_FNS))
    fits_hdus = astropy.io.fits.HDUList()
    for case_index, test_case_fn in enumerate(test_case_definitions.TEST_CASE_FNS):
        uniform_deviate = galsim.UniformDeviate(RANDOM_SEED + case_index)
        test_case = test_case_definitions.GalSimTestCase()
        test_case_fn(test_case)
        for band_index in xrange(5):
            image = test_case.construct_image(band_index, uniform_deviate)
            galsim.fits.write(image, hdu_list=fits_hdus)
            add_header_to_hdu(fits_hdus[-1], test_case.get_fits_header(case_index, band_index))

    image_file_name = os.path.join('output', 'galsim_test_images.fits')
    save_multi_extension_fits(fits_hdus, image_file_name)

if __name__ == "__main__":
    main()
