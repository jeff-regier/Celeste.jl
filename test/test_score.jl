import Celeste.AccuracyBenchmark
import FITSIO
import WCS


@testset "test the distance calculation" begin
    wd = pwd()
    cd(datadir)
    # this field is pretty near a pole--high declination means fewer pixels
    # per arc second of right ascention
    run(`make RUN=6075 CAMCOL=2 FIELD=29`)
    high_fits = FITSIO.FITS("6075/2/29/frame-r-006075-2-0029.fits")
    cd(wd)

    # we're just loaded this image to use its wcs transform
    high_wcs = WCS.from_header(FITSIO.read_header(high_fits[1], String))[1]
    pt0_pix = [10; 10.]
    pt0 = WCS.pix_to_world(high_wcs, pt0_pix)
    # pt1 is 1 arc second to the right of pt0
    pt1 = pt0 + [1. / 3600, 0]

    exact_dist = norm(WCS.world_to_pix(high_wcs, pt1) - WCS.world_to_pix(high_wcs, pt0))
    our_dist = AccuracyBenchmark.sky_distance_px(pt0..., pt1...)

    # both distances are about 0.315 pixels -- an arc second shift to the right
    # is less, in pixels, at higher elevations than at the equator, where it's
    # 1/0.396 ≈ 2.525 pixels per arc second.
    @test exact_dist ≈ our_dist atol=1e-4
end
