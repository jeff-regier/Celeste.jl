import logging

import generate_test_image

# Test cases follow. Each is a function which accepts a test case and adds LightSources and/or sets
# parameters. Each must be decorated with @generate_test_image.galsim_test_case. Function name will
# be included in the FITS header for reference.

@generate_test_image.galsim_test_case
def simple_star(test_case):
    test_case.add_star()

@generate_test_image.galsim_test_case
def star_position_1(test_case):
    test_case.add_star().offset_arcsec(-2, 0)

@generate_test_image.galsim_test_case
def star_position_2(test_case):
    test_case.add_star().offset_arcsec(0, 2)

@generate_test_image.galsim_test_case
def dim_star(test_case):
    test_case.add_star().reference_band_flux_nmgy(20)

@generate_test_image.galsim_test_case
def bright_star(test_case):
    test_case.add_star().reference_band_flux_nmgy(80)

@generate_test_image.galsim_test_case
def different_color_star(test_case):
    test_case.add_star().flux_relative_to_reference_band([0.2, 0.8, 1, 1.6, 1.3])

@generate_test_image.galsim_test_case
def star_with_noise(test_case):
    test_case.add_star().offset_arcsec(-1, 1).reference_band_flux_nmgy(20)
    test_case.sky_level_nmgy = 0.1
    test_case.include_noise = True

@generate_test_image.galsim_test_case
def angle_and_axis_ratio_1(test_case):
    test_case.add_galaxy().angle_deg(15).minor_major_axis_ratio(0.2)

@generate_test_image.galsim_test_case
def angle_and_axis_ratio_2(test_case):
    test_case.add_galaxy().angle_deg(160).minor_major_axis_ratio(0.4)

@generate_test_image.galsim_test_case
def round_galaxy(test_case):
    test_case.add_galaxy().minor_major_axis_ratio(1)

@generate_test_image.galsim_test_case
def small_galaxy(test_case):
    test_case.add_galaxy().half_light_radius_arcsec(0.75)

@generate_test_image.galsim_test_case
def large_galaxy(test_case):
    test_case.add_galaxy().half_light_radius_arcsec(2.5)

@generate_test_image.galsim_test_case
def dim_galaxy(test_case):
    test_case.add_galaxy().reference_band_flux_nmgy(5)

@generate_test_image.galsim_test_case
def bright_galaxy(test_case):
    test_case.add_galaxy().reference_band_flux_nmgy(20)

@generate_test_image.galsim_test_case
def different_color_galaxy(test_case):
    test_case.add_galaxy().flux_relative_to_reference_band([0.6, 0.2, 1, 1.1, 2])

@generate_test_image.galsim_test_case
def galaxy_with_all(test_case):
    (test_case.add_galaxy()
         .offset_arcsec(0.3, -0.7)
         .angle_deg(15)
         .minor_major_axis_ratio(0.4)
         .half_light_radius_arcsec(2.5)
         .reference_band_flux_nmgy(15)
    )

@generate_test_image.galsim_test_case
def galaxy_with_noise(test_case):
    galaxy_with_all(test_case)
    test_case.include_noise = True

@generate_test_image.galsim_test_case
def galaxy_with_low_background(test_case):
    galaxy_with_noise(test_case)
    test_case.sky_level_nmgy = 0.1

@generate_test_image.galsim_test_case
def galaxy_with_high_background(test_case):
    galaxy_with_noise(test_case)
    test_case.sky_level_nmgy = 0.3

@generate_test_image.galsim_test_case
def overlapping_stars(test_case):
    test_case.add_star().offset_arcsec(-3, 0)
    test_case.add_star().offset_arcsec(3, 0)

@generate_test_image.galsim_test_case
def overlapping_galaxies(test_case):
    test_case.add_galaxy().offset_arcsec(-2, -2).angle_deg(135).minor_major_axis_ratio(0.2)
    test_case.add_galaxy().offset_arcsec(3, 3).angle_deg(35).minor_major_axis_ratio(0.5)

@generate_test_image.galsim_test_case
def overlapping_star_and_galaxy(test_case):
    test_case.add_star().offset_arcsec(-5, 0)
    test_case.add_galaxy().offset_arcsec(2, 2).angle_deg(35).minor_major_axis_ratio(0.5)

@generate_test_image.galsim_test_case
def three_sources_two_overlap(test_case):
    test_case.add_star().offset_arcsec(-5, 5)
    (test_case.add_galaxy()
         .offset_arcsec(2, 5)
         .angle_deg(35)
         .minor_major_axis_ratio(0.2)
    )
    test_case.add_star().offset_arcsec(10, -10)

@generate_test_image.galsim_test_case
def three_sources_all_overlap(test_case):
    overlapping_star_and_galaxy(test_case)
    test_case.add_star().offset_arcsec(8, -1)

@generate_test_image.galsim_test_case
def smaller_psf(test_case):
    test_case.psf_sigma_pixels = 2
    test_case.add_star()

@generate_test_image.galsim_test_case
def larger_psf(test_case):
    test_case.psf_sigma_pixels = 6
    test_case.add_star()

def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    generate_test_image.generate_fits_file('galsim_benchmarks')

if __name__ == "__main__":
    main()
