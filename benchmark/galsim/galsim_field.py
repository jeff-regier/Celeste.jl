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
PSF_SIGMA_PX = 2.29 # similar to SDSS
BAND_SKY_LEVEL_NMGY = [0.2696, 0.3425, 0.7748, 1.6903, 4.9176]
BAND_NELEC_PER_NMGY = [146.9, 838.1, 829.8, 597.2, 129.8]

class MissingFieldError(Exception): pass

def set_image_dimensions(test_case, catalog_rows):
    min_ra_deg = min(float(row['ra']) for row in catalog_rows)
    max_ra_deg = max(float(row['ra']) for row in catalog_rows)
    min_dec_deg = min(float(row['dec']) for row in catalog_rows)
    max_dec_deg = max(float(row['dec']) for row in catalog_rows)

    height_arcsec = (max_ra_deg - min_ra_deg) * ARCSEC_PER_DEGREE + 2 * FIELD_EXPAND_ARCSEC
    width_arcsec = (max_dec_deg - min_dec_deg) * ARCSEC_PER_DEGREE + 2 * FIELD_EXPAND_ARCSEC
    arcsec_per_pixel = test_case.get_resolution()
    width_px = int(width_arcsec / arcsec_per_pixel)
    height_px = int(height_arcsec / arcsec_per_pixel)

    print('  Image dimensions {} W x {} H px'.format(width_px, height_px))
    test_case.set_dimensions(width_px, height_px)
    test_case.set_world_origin(
        min_ra_deg - FIELD_EXPAND_ARCSEC / ARCSEC_PER_DEGREE,
        min_dec_deg - FIELD_EXPAND_ARCSEC / ARCSEC_PER_DEGREE,
    )

def generate_field(test_case, catalog_csv):
    with open(catalog_csv) as stream:
        catalog_rows = list(csv.DictReader(stream))

    test_case.band_sky_level_nmgy = BAND_SKY_LEVEL_NMGY
    test_case.set_band_nelec_per_nmgy(BAND_NELEC_PER_NMGY)
    test_case.psf_sigma_pixels = PSF_SIGMA_PX
    set_image_dimensions(test_case, catalog_rows)

    for source_row in catalog_rows:
        def field(name):
            raw_value = source_row[name]
            try:
                return float(raw_value)
            except ValueError:
                raise MissingFieldError()

        try:
            colors = [
                field('color_ug'),
                field('color_gr'),
                field('color_ri'),
                field('color_iz'),
            ]
        except MissingFieldError:
            continue # just skip sources with missing colors

        if source_row['is_star'] == 'true':
            source = test_case.add_star()
        else:
            source = (
                test_case.add_galaxy()
                .half_light_radius_arcsec(field('gal_radius_px') * test_case.get_resolution())
                .gal_angle_deg(field('gal_angle_deg'))
                .axis_ratio(field('axis_ratio'))
                .gal_frac_dev(field('gal_frac_dev'))
            )
        source.world_coordinates_deg(field('ra'), field('dec'))
        source.flux_r_nmgy(field('flux_r_nmgy'))
        source.flux_relative_to_reference_band([
            math.exp(-colors[0] - colors[1]),
            math.exp(-colors[1]),
            1,
            math.exp(colors[2]),
            math.exp(colors[2] + colors[3]),
        ])

    test_case.include_noise = True

def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('catalog_csv')
    args = arg_parser.parse_args()

    def celeste_field_test(test_case):
        generate_field(test_case, args.catalog_csv)

    output_label = os.path.splitext(os.path.basename(args.catalog_csv))[0] + '_images'
    generate_test_image.generate_fits_file(output_label, [celeste_field_test])

if __name__ == "__main__":
    main()
