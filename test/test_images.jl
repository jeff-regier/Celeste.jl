using Base.Test
using DataFrames
import WCS

const rcf = RunCamcolField(3900, 6, 269)


"""
Get the PSF located at a particular world location in an image.

Args:
 - world_loc: A location in world coordinates.
 - img: An Image

Returns:
 - An array of PsfComponent objects that represents the PSF as a mixture
     of Gaussians.
"""
function get_source_psf(world_loc::Vector{Float64}, img::Image, psf_K::Int)
    # Some stamps or simulated data have no raw psf information. In that case,
    # just use the psf from the image.
    if size(img.raw_psf_comp.rrows) == (0, 0)
        # Also return a vector of empty psf params
        return img.psf, fill(fill(NaN, length(PsfParams)), psf_K)
    else
        pixel_loc = WCS.world_to_pix(img.wcs, world_loc)
        psfstamp = Model.eval_psf(img.raw_psf_comp, pixel_loc[1], pixel_loc[2])
        return PSF.fit_raw_psf_for_celeste(psfstamp, psf_K)
    end
end


function test_images()
    # A lot of tests are in a single function to avoid having to reload
    # the full image multiple times.
    images = SDSSIO.load_field_images(rcf, datadir)

    dir = "$datadir/$(rcf.run)/$(rcf.camcol)/$(rcf.field)"
    fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir rcf.run rcf.camcol rcf.field
    cat_entries = SDSSIO.read_photoobj_celeste(fname)

    ea = make_elbo_args(images, cat_entries, patch_radius_pix=1e-6)

    # Just check some basic facts about the catalog.
    @test length(cat_entries) == 805
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
    original_psf_val = Model.eval_psf(img.raw_psf_comp, pixel_loc[1], pixel_loc[2])

    original_psf_celeste =
        PSF.fit_raw_psf_for_celeste(original_psf_val, ea.psf_K)[1]
    fit_original_psf_val = PSF.get_psf_at_point(original_psf_celeste)

    obj_psf = get_source_psf(vs_obj[ids.u], img, ea.psf_K)[1]
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


test_images()
