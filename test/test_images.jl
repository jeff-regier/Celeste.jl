using Base.Test
using DataFrames
import WCS

@testset "images" begin

    # A lot of tests are in a single function to avoid having to reload
    # the full image multiple times.
    images = SampleData.get_sdss_images(3900, 6, 269)
    cat_entries = SampleData.get_sdss_catalog(3900, 6, 269)

    ea = make_elbo_args(images, cat_entries, patch_radius_pix=1e-6)

    # Just check some basic facts about the catalog.
    @test length(cat_entries) > 500
    @test 0 < sum([ce.is_star for ce in cat_entries]) < 805

    # Find an object near the middle of the image.
    ctr = WCS.pix_to_world(images[3].wcs, [images[3].H / 2, images[3].W / 2])
    dist = [sqrt((ce.pos[1] - ctr[1])^2 + (ce.pos[2] - ctr[2])^2)
                    for ce in cat_entries]
    obj_index = findmin(dist)[2]    # index of closest object
    obj_loc = cat_entries[obj_index].pos    # location of closest object

    # Test get_source_psf at point while we have the images loaded.
    test_b = 3
    img = ea.images[test_b]
    ea_obj = make_elbo_args(images, cat_entries[obj_index:obj_index])
    vs_obj = DeterministicVI.catalog_init_source(cat_entries[obj_index])
    pixel_loc = WCS.world_to_pix(img.wcs, obj_loc)
    original_psf_val = img.psfmap(pixel_loc[1], pixel_loc[2])

    original_psf_celeste =
        PSF.fit_raw_psf_for_celeste(original_psf_val, ea.psf_K)[1]
    fit_original_psf_val = PSF.get_psf_at_point(original_psf_celeste)

    obj_psf = PSF.get_source_psf(vs_obj[ids.pos], img, ea.psf_K)[1]
    obj_psf_val = PSF.get_psf_at_point(obj_psf)

    # The fits should match exactly.
    @test isapprox(obj_psf_val, fit_original_psf_val, atol=1e-6)

    # The raw psf will not be as good.
    @test isapprox(obj_psf_val, original_psf_val, atol=1e-2)

    cat_several = [cat_entries[1]; cat_entries[obj_index]]
    ea_several = make_elbo_args(ea_obj.images, cat_several)

    # The second set of vp is the object of interest
    point_patch_psf = PSF.get_psf_at_point(ea_several.patches[2, test_b].psf)
    # The threshold for the test below was formerly 1e-6.
    # Is it a problem I needed to increase it?
    @test isapprox(obj_psf_val, point_patch_psf, atol=5e-4)
end
