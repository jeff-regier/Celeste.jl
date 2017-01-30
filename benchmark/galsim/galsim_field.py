import logging
import os

import astropy.io.fits

import generate_test_image

FITS_CATALOG_FILENAME = os.path.join('output', 'galsim_field_500_catalog.fits')

@generate_test_image.galsim_test_case
def field_500(test_case):
    # width and height from SDSS (http://www.sdss.org/dr13/scope/)
    WIDTH_PX = 1361
    HEIGHT_PX = 2048
    test_case.set_dimensions(WIDTH_PX, HEIGHT_PX)
    image_dimensions_arcsec = test_case.get_dimensions_arcsec()

    hdu_list = astropy.io.fits.open(FITS_CATALOG_FILENAME)
    try:
        for source_row in hdu_list[1].data:
            if source_row.field('is_star'):
                source = test_case.add_star()
            else:
                source = (
                    test_case.add_galaxy()
                    .half_light_radius_arcsec(source_row.field('half_light_radius_arcsec'))
                    .angle_deg(source_row.field('angle_deg'))
                    .minor_major_axis_ratio(source_row.field('minor_major_axis_ratio'))
                    .de_vaucouleurs_mixture_weight(source_row.field('de_vaucouleurs_mixture_weight'))
                )
            source.offset_arcsec(
                (source_row.field('relative_position_x') - 0.5) * image_dimensions_arcsec[0],
                (source_row.field('relative_position_y') - 0.5) * image_dimensions_arcsec[1],
            )
            source.reference_band_flux_nmgy(source_row.field('reference_band_flux_nmgy'))
            source.flux_relative_to_reference_band([
                1 / (source_row.field('color_ratio_21') * source_row.field('color_ratio_32')),
                1 / source_row.field('color_ratio_32'),
                1,
                source_row.field('color_ratio_43'),
                source_row.field('color_ratio_43') * source_row.field('color_ratio_54'),
            ])
    finally:
        hdu_list.close()

    test_case.include_noise = True

def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    generate_test_image.generate_fits_file('galsim_field')

if __name__ == "__main__":
    main()
