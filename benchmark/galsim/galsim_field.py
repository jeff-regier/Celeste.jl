import argparse
import csv
import logging
import math
import os

import astropy.io.fits

import generate_test_image

FITS_CATALOG_FILENAME = os.path.join('output', 'galsim_field_500_catalog.fits')
ARCSEC_PER_DEGREE = 3600.
FIELD_EXPAND_ARCSEC = 10

def set_image_dimensions(test_case, catalog_rows):
    min_ra_deg = min(float(row['right_ascension_deg']) for row in catalog_rows)
    max_ra_deg = max(float(row['right_ascension_deg']) for row in catalog_rows)
    min_dec_deg = min(float(row['declination_deg']) for row in catalog_rows)
    max_dec_deg = max(float(row['declination_deg']) for row in catalog_rows)

    width_arcsec = (max_ra_deg - min_ra_deg) * ARCSEC_PER_DEGREE + 2 * FIELD_EXPAND_ARCSEC
    height_arcsec = (max_dec_deg - min_dec_deg) * ARCSEC_PER_DEGREE + 2 * FIELD_EXPAND_ARCSEC
    arcsec_per_pixel = test_case.get_resolution()
    width_px = width_arcsec / arcsec_per_pixel
    height_px = height_arcsec / arcsec_per_pixel

    print('  Image dimensions {} x {} px'.format(width_px, height_px))
    test_case.set_dimensions(width_px, height_px)
    test_case.set_world_origin(
        min_ra_deg - FIELD_EXPAND_ARCSEC / ARCSEC_PER_DEGREE,
        min_dec_deg - FIELD_EXPAND_ARCSEC / ARCSEC_PER_DEGREE,
    )

def generate_field(test_case, catalog_csv):
    with open(catalog_csv) as stream:
        catalog_rows = list(csv.DictReader(stream))

    set_image_dimensions(test_case, catalog_rows)

    for source_row in catalog_rows:
        def field(name): return float(source_row[name])
        if source_row['is_star'] == 'true':
            source = test_case.add_star()
        else:
            source = (
                test_case.add_galaxy()
                .half_light_radius_arcsec(field('half_light_radius_px') * test_case.get_resolution())
                .angle_deg(field('angle_deg'))
                .minor_major_axis_ratio(field('minor_major_axis_ratio'))
                .de_vaucouleurs_mixture_weight(field('de_vaucouleurs_mixture_weight'))
            )
        source.world_coordinates_deg(field('right_ascension_deg'), field('declination_deg'))
        source.reference_band_flux_nmgy(field('reference_band_flux_nmgy'))
        source.flux_relative_to_reference_band([
            math.exp(-field('color_log_ratio_ug') - field('color_log_ratio_gr')),
            math.exp(-field('color_log_ratio_gr')),
            1,
            math.exp(field('color_log_ratio_ri')),
            math.exp(field('color_log_ratio_ri') + field('color_log_ratio_iz')),
        ])

    test_case.include_noise = True

def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('catalog_csv')
    args = arg_parser.parse_args()

    def celeste_field_test(test_case):
        generate_field(test_case, args.catalog_csv)

    generate_test_image.generate_fits_file('galsim_field', [celeste_field_test])

if __name__ == "__main__":
    main()
