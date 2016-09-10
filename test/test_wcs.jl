import DataFrames
import FITSIO
import WCS


const band_letters = ['u', 'g', 'r', 'i', 'z']


"Test that the identity WCSTransform works as expected."
function test_id_wcs()
    rand_coord = rand(2, 10)
    @test WCS.pix_to_world(SampleData.wcs_id, rand_coord) == rand_coord
    @test WCS.world_to_pix(SampleData.wcs_id, rand_coord) == rand_coord
end


function test_linear_world_to_pix()
    rcf = SDSSIO.RunCamcolField(3900, 6, 269)
    image = SDSSIO.load_field_images(rcf, datadir)[1]
    wcs = image.wcs

    pix_center = Float64[0.5 * image.H, 0.5 * image.W]
    pix_loc = pix_center + [5., 3.]
    world_center = WCS.pix_to_world(wcs, pix_center)
    world_loc = WCS.pix_to_world(wcs, pix_loc)

    function test_jacobian(wcs, pix_center, world_center)
        wcs_jacobian = Model.pixel_world_jacobian(wcs, pix_center);

        pix_loc_test1 = WCS.world_to_pix(wcs, world_loc)
        pix_loc_test2 = Model.linear_world_to_pix(wcs_jacobian, world_center, pix_center,
                                       world_loc)

        # Note that the accuracy of the linear approximation isn't great.
        @test_approx_eq(pix_loc_test1, pix_loc)
        @test_approx_eq_eps(pix_loc_test2, pix_loc, 1e-2)
    end

    @test Model.pixel_world_jacobian(SampleData.wcs_id, pix_center) == [1.0 0.0; 0.0 1.0];

    test_jacobian(wcs, pix_center, world_center)
end


test_id_wcs()
test_linear_world_to_pix()
