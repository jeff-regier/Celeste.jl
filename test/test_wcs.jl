println("Running WCSUtils tests.")

import WCS: world_to_pix, pix_to_world


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
    wcs = SDSS.load_raw_field(datadir, run_num, camcol_num, field_num, 1, 1.0)[7];

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
        SDSS.load_raw_field(datadir, run_num, camcol_num, field_num, 3, 1.0);

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
