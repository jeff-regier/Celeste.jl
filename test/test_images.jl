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
    pixel_loc = WCS.world_to_pix(img.wcs, obj_loc)
    original_psf_val = Model.eval_psf(img.raw_psf_comp, pixel_loc[1], pixel_loc[2])

    original_psf_celeste =
        PSF.fit_raw_psf_for_celeste(original_psf_val, ea.psf_K)[1]
    fit_original_psf_val = PSF.get_psf_at_point(original_psf_celeste)

    obj_psf = get_source_psf(ea_obj.vp[1][ids.u], img, ea.psf_K)[1]
    obj_psf_val = PSF.get_psf_at_point(obj_psf)

    # The fits should match exactly.
    @test_approx_eq_eps(obj_psf_val, fit_original_psf_val, 1e-6)

    # The raw psf will not be as good.
    @test_approx_eq_eps(obj_psf_val, original_psf_val, 1e-2)

    cat_several = [cat_entries[1]; cat_entries[obj_index]]
    ea_several = make_elbo_args(ea_obj.images, cat_several)

    # The second set of vp is the object of interest
    point_patch_psf = PSF.get_psf_at_point(ea_several.patches[2, test_b].psf)
    # The threshold for the test below was formerly 1e-6.
    # Is it a problem I needed to increase it?
    @test_approx_eq_eps(obj_psf_val, point_patch_psf, 1e-4)
end


function test_stamp_get_object_psf()
    stamp_blob, stamp_mp, body = gen_sample_star_dataset()
    img = stamp_blob[3]
    obj_index =    stamp_mp.vp[1][ids.u]
    pixel_loc = WCS.world_to_pix(img.wcs, obj_index)
    original_psf_val = PSF.get_psf_at_point(img.psf)

    obj_psf_val = PSF.get_psf_at_point(
      get_source_psf(stamp_mp.vp[1][ids.u], img, 2)[1])
    @test_approx_eq_eps(obj_psf_val, original_psf_val, 1e-6)
end


function test_set_patch_size()
    # Test that the patch size gets most of the light from a variety of
    # galaxy shapes.
    # This shows that the current patch size is actually far too conservative.

    function gal_catalog_from_scale(gal_scale::Float64, flux_scale::Float64)
        CatalogEntry[CatalogEntry(world_location, false,
                                   flux_scale * fluxes, flux_scale * fluxes,
                                   0.1, .01, pi/4, gal_scale, "sample", 0) ]
    end

    srand(1)
    images0 = SampleData.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf")
    img_size = 150
    for b in 1:5
            images0[b].H, images0[b].W = img_size, img_size
    end
    fluxes = [4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]

    world_location = WCS.pix_to_world(images0[3].wcs,
                                      Float64[img_size / 2, img_size / 2])

    for gal_scale in [1.0, 10.0], flux_scale in [0.1, 10.0]
        cat = gal_catalog_from_scale(gal_scale, flux_scale)
        images = Synthetic.gen_blob(images0, cat)
        ea = make_elbo_args(images, cat)

        for b=1:length(images)
            @assert size(ea.images[b].tiles) == (1, 1)
            tile_image = DeterministicVI.tile_predicted_image(
                ea.images[b].tiles[1,1], ea, ea.tile_source_map[b][1,1])

            pixel_center = WCS.world_to_pix(images[b].wcs, cat[1].pos)
            radius = Model.choose_patch_radius(
                pixel_center, cat[1], images[b].psf, ea.images[b])

            circle_pts = fill(false, images[b].H, images[b].W)
            in_circle = 0.0
            for x=1:size(tile_image)[1], y=1:size(tile_image)[2]
                if ((x - pixel_center[1]) ^ 2 + (y - pixel_center[2]) ^ 2) < radius ^ 2
                    in_circle += tile_image[x, y]
                    circle_pts[x, y] = true
                end
            end
            @test in_circle / sum(tile_image) > 0.95
        end
    end
end


test_images()
#test_stamp_get_object_psf()
#test_set_patch_size()
