import logging
import os

import generate_test_image

OUTPUT_LABEL = 'galsim_benchmarks'

# populated by the `galsim_test_case` decorator
TEST_CASE_CALLBACKS = []

def galsim_test_case(fn):
    def decorated(test_case):
        fn(test_case)
        test_case.comment = fn.__name__
    decorated.__name__ = fn.__name__
    TEST_CASE_CALLBACKS.append(decorated)
    return decorated

# Test cases follow. Each is a function which accepts a test case and adds LightSources and/or sets
# parameters. Each must be decorated with @galsim_test_case. Function name will
# be included in the FITS header for reference.

@galsim_test_case
def simple_star(test_case):
    test_case.add_star()

@galsim_test_case
def star_position_1(test_case):
    test_case.add_star().offset_arcsec(-2, 0)

@galsim_test_case
def star_position_2(test_case):
    test_case.add_star().offset_arcsec(0, 2)

@galsim_test_case
def dim_star(test_case):
    test_case.add_star().flux_r_nmgy(20)

@galsim_test_case
def bright_star(test_case):
    test_case.add_star().flux_r_nmgy(80)

@galsim_test_case
def different_color_star(test_case):
    test_case.add_star().flux_relative_to_reference_band([0.2, 0.8, 1, 1.6, 1.3])

@galsim_test_case
def star_with_noise(test_case):
    test_case.add_star().offset_arcsec(-1, 1).flux_r_nmgy(20)
    test_case.sky_level_nmgy = 0.1
    test_case.include_noise = True

@galsim_test_case
def angle_and_axis_ratio_1(test_case):
    test_case.add_galaxy().gal_angle_deg(15).axis_ratio(0.2)

@galsim_test_case
def angle_and_axis_ratio_2(test_case):
    test_case.add_galaxy().gal_angle_deg(160).axis_ratio(0.4)

@galsim_test_case
def round_galaxy(test_case):
    test_case.add_galaxy().axis_ratio(1)

@galsim_test_case
def small_galaxy(test_case):
    test_case.add_galaxy().half_light_radius_arcsec(0.75)

@galsim_test_case
def large_galaxy(test_case):
    test_case.add_galaxy().half_light_radius_arcsec(2.5)

@galsim_test_case
def dim_galaxy(test_case):
    test_case.add_galaxy().flux_r_nmgy(5)

@galsim_test_case
def bright_galaxy(test_case):
    test_case.add_galaxy().flux_r_nmgy(20)

@galsim_test_case
def de_vaucouleurs_galaxy(test_case):
    test_case.add_galaxy().gal_frac_dev(1)

@galsim_test_case
def exp_dev_mixture_galaxy(test_case):
    test_case.add_galaxy().gal_frac_dev(0.4)

@galsim_test_case
def different_color_galaxy(test_case):
    test_case.add_galaxy().flux_relative_to_reference_band([0.6, 0.2, 1, 1.1, 2])

@galsim_test_case
def galaxy_with_all(test_case):
    (test_case.add_galaxy()
         .offset_arcsec(0.3, -0.7)
         .gal_angle_deg(15)
         .axis_ratio(0.4)
         .half_light_radius_arcsec(2.5)
         .flux_r_nmgy(15)
         .gal_frac_dev(0.4)
         .flux_relative_to_reference_band([0.6, 0.2, 1, 1.1, 2])
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
    test_case.add_star().offset_arcsec(-3, 0)
    test_case.add_star().offset_arcsec(3, 0)

@galsim_test_case
def overlapping_galaxies(test_case):
    test_case.add_galaxy().offset_arcsec(-2, -2).gal_angle_deg(135).axis_ratio(0.2)
    test_case.add_galaxy().offset_arcsec(3, 3).gal_angle_deg(35).axis_ratio(0.5)

@galsim_test_case
def overlapping_star_and_galaxy(test_case):
    test_case.add_star().offset_arcsec(-5, 0)
    test_case.add_galaxy().offset_arcsec(2, 2).gal_angle_deg(35).axis_ratio(0.5)

@galsim_test_case
def three_sources_two_overlap(test_case):
    test_case.add_star().offset_arcsec(-5, 5)
    (test_case.add_galaxy()
         .offset_arcsec(2, 5)
         .gal_angle_deg(35)
         .axis_ratio(0.2)
    )
    test_case.add_star().offset_arcsec(10, -10)

@galsim_test_case
def three_sources_all_overlap(test_case):
    overlapping_star_and_galaxy(test_case)
    test_case.add_star().offset_arcsec(8, -1)

@galsim_test_case
def smaller_psf(test_case):
    test_case.psf_sigma_pixels = 2
    test_case.add_star()

@galsim_test_case
def larger_psf(test_case):
    test_case.psf_sigma_pixels = 6
    test_case.add_star()

def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    final_filename = generate_test_image.generate_fits_file(OUTPUT_LABEL, TEST_CASE_CALLBACKS)
    generate_test_image.write_latest_filename(OUTPUT_LABEL, os.path.basename(final_filename))

if __name__ == "__main__":
    main()
