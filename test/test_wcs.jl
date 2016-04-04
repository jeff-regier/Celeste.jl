println("Running WCSUtils tests.")

import WCS: world_to_pix, pix_to_world

import DataFrames
import FITSIO
import Grid
import WCS


const band_letters = ['u', 'g', 'r', 'i', 'z']


"""
oad the raw electron counts, calibration vector, and sky background from a field.

Args:
 - field_dir: The directory of the file
 - run_num: The run number
 - camcol_num: The camcol number
 - field_num: The field number
 - b: The filter band (a number from 1 to 5)
 - gain: The gain for this band (e.g. as read from photoField)

Returns:
 - nelec: An image of raw electron counts in nanomaggies
 - calib_col: A column of calibration values (the same for every column of the image)
 - sky_grid: A CoordInterpGrid bilinear interpolation object
 - sky_x: The x coordinates at which to evaluate sky_grid to match nelec.
 - sky_y: The y coordinates at which to evaluate sky_grid to match nelec.
 - sky_image: The sky interpolated to the original image size.
 - wcs: A WCS.WCSTransform object for convert between world and pixel
   coordinates.

The meaing of the frame data structures is thoroughly documented here:
http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
"""
function load_raw_field(field_dir, run_num, camcol_num, field_num, b, gain)
    @assert 1 <= b <= 5
    b_letter = band_letters[b]

    img_filename = "$field_dir/frame-$b_letter-$run_num-$camcol_num-$field_num.fits"
    img_fits = FITSIO.FITS(img_filename)
    @assert length(img_fits) == 4

    # This is the sky-subtracted and calibrated image.
    processed_image = read(img_fits[1])

    # Read in the sky background.
    sky_image_raw = read(img_fits[3], "ALLSKY")
    sky_x = collect(read(img_fits[3], "XINTERP"))
    sky_y = collect(read(img_fits[3], "YINTERP"))

    # Get the WCS coordinates.
    header_str = FITSIO.read_header(img_fits[1], ASCIIString)
    wcs = WCS.from_header(header_str)[1]

    # These are the column types (not currently used).
    ctype = [FITSIO.read_key(img_fits[1], "CTYPE1")[1],
             FITSIO.read_key(img_fits[1], "CTYPE2")[1]]

    # This is the calibration vector:
    calib_col = read(img_fits[2])
    calib_image = [ calib_col[row] for
                    row in 1:size(processed_image)[1],
                    col in 1:size(processed_image)[2] ]

    close(img_fits)

    # Interpolate the sky to the full image.  Combining the example from
    # http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
    # ...with the documentation from the IDL language:
    # http://www.exelisvis.com/docs/INTERPOLATE.html
    # ...we can see that we are supposed to interpret a point (sky_x[i], sky_y[j])
    # with associated row and column (if, jf) = (floor(sky_x[i]), floor(sky_y[j]))
    # as lying in the square spanned by the points
    # (sky_image_raw[if, jf], sky_image_raw[if + 1, jf + 1]).
    # ...keeping in mind that IDL uses zero indexing:
    # http://www.exelisvis.com/docs/Manipulating_Arrays.html
    sky_grid_vals = ((1:1.:size(sky_image_raw)[1]) - 1, (1:1.:size(sky_image_raw)[2]) - 1)
    sky_grid = Grid.CoordInterpGrid(sky_grid_vals, sky_image_raw[:,:,1],
                                    Grid.BCnearest, Grid.InterpLinear)

    # This interpolation is really slow.
    sky_image = [ sky_grid[x, y] for x in sky_x, y in sky_y ]

    # Convert to raw electron counts.  Note that these may not be close to integers
    # due to the analog to digital conversion process in the telescope.
    nelec = gain * convert(Array{Float64, 2}, (processed_image ./ calib_image .+ sky_image))

    nelec, calib_col, sky_grid, sky_x, sky_y, sky_image, wcs
end

"Test that the identity WCSTransform works as expected."
function test_id_wcs()
    rand_coord = rand(2, 10)
    @test pix_to_world(WCSUtils.wcs_id, rand_coord) == rand_coord
    @test world_to_pix(WCSUtils.wcs_id, rand_coord) == rand_coord
end


function test_pixel_deriv_to_world_deriv()
    run_num = "003900"
    camcol_num = "6"
    field_num = "0269"

    # The gain is wrong but it doesn't matter.
    wcs = load_raw_field(datadir, run_num, camcol_num, field_num, 1, 1.0)[7];

    function test_fun(pix_loc::Array{Float64, 1})
        pix_loc[1]^2 + 0.5 * pix_loc[2]
    end

    function test_fun_grad(pix_loc::Array{Float64, 1})
        Float64[2 * pix_loc[1], 0.5 ]
    end

    function test_fun_world(world_loc::Array{Float64, 1}, wcs)
        pix_loc = WCSUtils.world_to_pix(wcs, world_loc)
        test_fun(pix_loc)
    end

    pix_del = 1e-3
    world_del = 1e-9
    pix_loc = Float64[5, 5]
    pix_loc_1 = pix_loc + pix_del * [1, 0]
    pix_loc_2 = pix_loc + pix_del * [0, 1]
    world_loc = pix_to_world(wcs, pix_loc)
    world_loc_1 = world_loc + world_del * [1, 0]
    world_loc_2 = world_loc + world_del * [0, 1]

    @test_approx_eq_eps test_fun(pix_loc) test_fun_world(world_loc, wcs) 1e-8

    pix_deriv = test_fun_grad(pix_loc)
    world_deriv = Float64[ (test_fun_world(world_loc_1, wcs) -
                            test_fun_world(world_loc, wcs)) / world_del
                           (test_fun_world(world_loc_2, wcs) -
                            test_fun_world(world_loc, wcs)) / world_del ]

    relative_err = (WCSUtils.pixel_deriv_to_world_deriv(wcs, pix_deriv, pix_loc) -
                    world_deriv) ./ abs(world_deriv)
    @test_approx_eq_eps relative_err [ 0 0 ] 1e-3
end


function test_world_to_pix()
    run_num = "003900"
    camcol_num = "6"
    field_num = "0269"

    # The gain will not be used.
    nelec, calib_col, sky_grid, sky_x, sky_y, sky_image, wcs =
        load_raw_field(datadir, run_num, camcol_num, field_num, 3, 1.0);

    pix_center = Float64[0.5 * size(nelec, 1), 0.5 * size(nelec, 1)]
    pix_loc = pix_center + [5., 3.]
    world_center = pix_to_world(wcs, pix_center)
    world_loc = pix_to_world(wcs, pix_loc)

    function test_jacobian(wcs, pix_center, world_center)
      wcs_jacobian = WCSUtils.pixel_world_jacobian(wcs, pix_center);

      pix_loc_test1 = world_to_pix(wcs, world_loc)
      pix_loc_test2 = world_to_pix(wcs_jacobian, world_center, pix_center,
                                   world_loc)

      # Note that the accuracy of the linear approximation isn't great.
      @test_approx_eq(pix_loc_test1, pix_loc)
      @test_approx_eq_eps(pix_loc_test2, pix_loc, 1e-2)
    end

    @test WCSUtils.pixel_world_jacobian(WCSUtils.wcs_id, pix_center) == [1.0 0.0; 0.0 1.0];

    test_jacobian(wcs, pix_center, world_center)
end


test_id_wcs()
test_pixel_deriv_to_world_deriv()
test_world_to_pix()
