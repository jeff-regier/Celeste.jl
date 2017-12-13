import DataFrames
import FITSIO
import WCS
using Celeste.SDSSIO

@testset "wcs" begin
    @testset "the identity WCSTransform works as expected" begin
        rand_coord = rand(2, 10)
        @test WCS.pix_to_world(SampleData.wcs_id, rand_coord) == rand_coord
        @test WCS.world_to_pix(SampleData.wcs_id, rand_coord) == rand_coord
    end

    @testset "linear_world_to_pix works" begin
        image = SampleData.get_sdss_images(3900, 6, 269)[1]
        wcs = image.wcs

        pix_center = Float64[0.5 * image.H, 0.5 * image.W]
        pix_loc = pix_center + [5., 3.]
        world_center = WCS.pix_to_world(wcs, pix_center)
        world_loc = WCS.pix_to_world(wcs, pix_loc)

        @test (Model.pixel_world_jacobian(SampleData.wcs_id, pix_center) ==
               [1.0 0.0; 0.0 1.0])

        wcs_jacobian = Model.pixel_world_jacobian(wcs, pix_center);

        pix_loc_test1 = WCS.world_to_pix(wcs, world_loc)
        pix_loc_test2 = Model.linear_world_to_pix(wcs_jacobian, world_center,
                                                  pix_center, world_loc)

        # Note that the accuracy of the linear approximation isn't great.
        @test pix_loc_test1 ≈ pix_loc
        @test isapprox(pix_loc_test2, pix_loc, atol=1e-2)
    end
end
