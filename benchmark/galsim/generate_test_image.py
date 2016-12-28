import hashlib
import logging
import os

import astropy.io.fits
import galsim

import test_case_definitions

_logger = logging.getLogger(__name__) 

RANDOM_SEED = 1234
LATEST_FITS_FILENAME_HOLDER = 'latest_fits_filename.txt'
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

def append_md5sum_to_filename(filename):
    md5 = hashlib.md5()
    with open(filename, 'rb') as stream:
        md5.update(stream.read())
    md5sum = md5.hexdigest()
    root, extension = os.path.splitext(filename)
    new_filename = '{}_{}{}'.format(root, md5.hexdigest()[:10], extension)
    os.rename(filename, new_filename)
    return new_filename

def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    _logger.info('Generating %d test cases', len(test_case_definitions.TEST_CASE_FNS))
    fits_hdus = astropy.io.fits.HDUList()
    for case_index, test_case_fn in enumerate(test_case_definitions.TEST_CASE_FNS):
        _logger.info('  Generating case %s', test_case_fn.__name__)
        uniform_deviate = galsim.UniformDeviate(RANDOM_SEED + case_index)
        test_case = test_case_definitions.GalSimTestCase()
        test_case_fn(test_case)
        for band_index in xrange(5):
            image = test_case.construct_image(band_index, uniform_deviate)
            galsim.fits.write(image, hdu_list=fits_hdus)
            add_header_to_hdu(fits_hdus[-1], test_case.get_fits_header(case_index, band_index))

    image_file_name = os.path.join('output', 'galsim_test_images.fits')
    save_multi_extension_fits(fits_hdus, image_file_name)
    final_filename = append_md5sum_to_filename(image_file_name)
    _logger.info('Wrote multi-extension FITS file to %r', final_filename)

    with open(LATEST_FITS_FILENAME_HOLDER, 'w') as stream:
        stream.write(os.path.basename(final_filename))
        stream.write('\n')
    _logger.info('Updated %r', LATEST_FITS_FILENAME_HOLDER)

if __name__ == "__main__":
    main()
