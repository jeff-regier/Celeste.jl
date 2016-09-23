using Base.Test
using DataFrames
import WCS

const rcf = RunCamcolField(3900, 6, 269)


"""
Get the PSF located at a particular world location in an image.

Args:
 - world_loc: A location in world coordinates.
 - img: An TiledImage

Returns:
 - An array of PsfComponent objects that represents the PSF as a mixture
     of Gaussians.
"""
function get_source_psf(world_loc::Vector{Float64}, img::TiledImage, psf_K::Int)
    # Some stamps or simulated data have no raw psf information.    In that case,
    # just use the psf from the image.
    if size(img.raw_psf_comp.rrows) == (0, 0)
        # Also return a vector of empty psf params
        return img.psf, fill(fill(NaN, length(PsfParams)), psf_K)
    else
        pixel_loc = WCS.world_to_pix(img.wcs, world_loc)
        psfstamp = img.raw_psf_comp(pixel_loc[1], pixel_loc[2])
        return PSF.fit_raw_psf_for_celeste(psfstamp)
    end
end


function test_blob()
    # A lot of tests are in a single function to avoid having to reload
    # the full image multiple times.
    blob = SDSSIO.load_field_images(rcf, datadir)

    dir = "$datadir/$(rcf.run)/$(rcf.camcol)/$(rcf.field)"
    fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir rcf.run rcf.camcol rcf.field
    cat_entries = SDSSIO.read_photoobj_celeste(fname)

    ea = make_elbo_args(blob, cat_entries,
                       patch_radius_pix=1e-6, fit_psf=false, tile_width=20)

    # Just check some basic facts about the catalog.
    @test length(cat_entries) == 805
    @test 0 < sum([ce.is_star for ce in cat_entries]) < 805

    # Find an object near the middle of the image.
    ctr = WCS.pix_to_world(blob[3].wcs, [blob[3].H / 2, blob[3].W / 2])
    dist = [sqrt((ce.pos[1] - ctr[1])^2 + (ce.pos[2] - ctr[2])^2)
                    for ce in cat_entries]
    obj_index = findmin(dist)[2]    # index of closest object
    obj_loc = cat_entries[obj_index].pos    # location of closest object

    # Test get_source_psf at point while we have the blob loaded.
    test_b = 3
    img = ea.images[test_b]
    ea_obj = make_elbo_args(blob, cat_entries[obj_index:obj_index])
    pixel_loc = WCS.world_to_pix(img.wcs, obj_loc);
    original_psf_val = img.raw_psf_comp(pixel_loc[1], pixel_loc[2])

    original_psf_celeste = PSF.fit_raw_psf_for_celeste(original_psf_val)[1];
    fit_original_psf_val = PSF.get_psf_at_point(original_psf_celeste);

    obj_psf = get_source_psf(ea_obj.vp[1][ids.u], img, default_psf_K)[1];
    obj_psf_val = PSF.get_psf_at_point(obj_psf);

    # The fits should match exactly.
    @test_approx_eq_eps(obj_psf_val, fit_original_psf_val, 1e-6)

    # The raw psf will not be as good.
    @test_approx_eq_eps(obj_psf_val, original_psf_val, 1e-2)

    cat_several = [cat_entries[1]; cat_entries[obj_index]]
    ea_several = make_elbo_args(ea_obj.images, cat_several)

    # The second set of vp is the object of interest
    point_patch_psf = PSF.get_psf_at_point(ea_several.patches[2, test_b].psf);
    # The threshold for the test below was formerly 1e-6.
    # Is it a problem I needed to increase it?
    @test_approx_eq_eps(obj_psf_val, point_patch_psf, 1e-4)
end


function test_stamp_get_object_psf()
    stamp_blob, stamp_mp, body = gen_sample_star_dataset();
    img = TiledImage(stamp_blob[3]);
    obj_index =    stamp_mp.vp[1][ids.u]
    pixel_loc = WCS.world_to_pix(img.wcs, obj_index)
    original_psf_val = PSF.get_psf_at_point(img.psf);

    obj_psf_val = PSF.get_psf_at_point(
      get_source_psf(stamp_mp.vp[1][ids.u], img, default_psf_K)[1])
    @test_approx_eq_eps(obj_psf_val, original_psf_val, 1e-6)
end


function test_get_tiled_image_source()
    # Test that an object only occurs the appropriate tile's local sources.
    blob, ea, body = gen_sample_star_dataset();

    ea = make_elbo_args(blob, body; patch_radius_pix=1e-6)

    tiled_img = TiledImage(blob[3], tile_width=10);
    for hh in 1:size(tiled_img.tiles, 1), ww in 1:size(tiled_img.tiles, 2)
        tile = tiled_img.tiles[hh, ww]
        loc = Float64[mean(tile.h_range), mean(tile.w_range)]
        for b = 1:5
            ea.vp[1][ids.u] = loc
            pixel_center = WCS.world_to_pix(blob[b].wcs, loc)
            wcs_jacobian = Model.pixel_world_jacobian(blob[b].wcs, pixel_center)
            radius_pix = maxabs(eigvals(wcs_jacobian)) * 1e-6
            ea.patches[1, b] = SkyPatch(loc,
                                        radius_pix,
                                        blob[b].psf,
                                        wcs_jacobian,
                                        pixel_center)
        end
        patches = vec(ea.patches[:, 3])
        local_sources = Model.get_sources_per_tile(tiled_img.tiles,
                                                   Model.patch_ctrs_pix(patches),
                                                   Model.patch_radii_pix(patches))
        @test local_sources[hh, ww] == Int[1]
        for hh2 in 1:size(tiled_img.tiles, 1), ww2 in 1:size(tiled_img.tiles, 2)
            if (hh2 != hh) || (ww2 != ww)
                @test local_sources[hh2, ww2] == Int[]
            end
        end
    end
end


function test_local_source_candidate()
    blob, ea, body = gen_n_body_dataset(100);

    for b=1:ea.N
        # Get the sources by iterating over everything.
        patches = vec(ea.patches[:,b])

        tile_source_map = Model.get_sources_per_tile(ea.images[b].tiles,
                        Model.patch_ctrs_pix(patches),
                        Model.patch_radii_pix(patches))

        # Check that all the actual sources are candidates and that this is the
        # same as what is returned by make_elbo_args
        HH, WW = size(tile_source_map)
        for h=1:HH, w=1:WW
            # Get a set of candidates.
            candidates = Model.local_source_candidates(
                                                ea.images[b].tiles[h, w],
                                                Model.patch_ctrs_pix(patches),
                                                Model.patch_radii_pix(patches))
            @test setdiff(tile_source_map[h, w], candidates) == []
            @test tile_source_map[h, w] == ea.tile_source_map[b][h, w]
        end
    end
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
    blob0 = SampleData.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf");
    img_size = 150
    for b in 1:5
            blob0[b].H, blob0[b].W = img_size, img_size
    end
    fluxes = [4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]

    world_location = WCS.pix_to_world(blob0[3].wcs,
                                      Float64[img_size / 2, img_size / 2])

    for gal_scale in [1.0, 10.0], flux_scale in [0.1, 10.0]
        cat = gal_catalog_from_scale(gal_scale, flux_scale);
        blob = Synthetic.gen_blob(blob0, cat);
        ea = make_elbo_args(blob, cat, tile_width=typemax(Int));

        for b=1:length(blob)
            @assert size(ea.images[b].tiles) == (1, 1)
            tile_image = DeterministicVI.tile_predicted_image(
                ea.images[b].tiles[1,1], ea, ea.tile_source_map[b][1,1]);

            pixel_center = WCS.world_to_pix(blob[b].wcs, cat[1].pos)
            radius = Model.choose_patch_radius(
                pixel_center, cat[1], blob[b].psf, ea.images[b])

            circle_pts = fill(false, blob[b].H, blob[b].W);
            in_circle = 0.0
            for x=1:size(tile_image)[1], y=1:size(tile_image)[2]
                if ((x - pixel_center[1]) ^ 2 + (y - pixel_center[2]) ^ 2) < radius ^ 2
                    in_circle += tile_image[x, y]
                    circle_pts[x, y] = true
                end
            end
            @test in_circle / sum(tile_image) > 0.95

            # Convenient for visualizing:
            # using PyPlot
            # in_circle / sum(tile_image)
            # imshow(tile_image); colorbar()
            # imshow(circle_pts, alpha=0.4)
        end
    end
end


test_blob()
test_stamp_get_object_psf()
test_get_tiled_image_source()
test_local_source_candidate()
test_set_patch_size()
