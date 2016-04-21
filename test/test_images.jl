using Base.Test
using DataFrames

import WCS

using Celeste: Model
import Celeste: ModelInit, ElboDeriv, SDSSIO, PSF

println("Running SkyImages tests.")

const RUN = 3900
const CAMCOL = 6
const FIELD = 269



"""
Crop an image in place to a (2 * width) x (2 * width) - pixel square centered
at the world coordinates wcs_center.
Args:
  - blob: The field to crop
  - width: The width in pixels of each quadrant
  - wcs_center: A location in world coordinates (e.g. the location of a
                celestial body)

Returns:
  - A tiled blob with a single tile in each image centered at wcs_center.
    This can be used to investigate a certain celestial object in a single
    tiled blob, for example.
"""
function crop_blob_to_location(
  blob::Array{Image, 1},
  width::Union{Float64, Int},
  wcs_center::Vector{Float64})
    @assert length(wcs_center) == 2
    @assert width > 0

    tiled_blob = Array(TiledImage, length(blob))
    for b=1:length(blob)
        # Get the pixels that are near enough to the wcs_center.
        pix_center = WCS.world_to_pix(blob[b].wcs, wcs_center)
        h_min = max(floor(Int, pix_center[1] - width), 1)
        h_max = min(ceil(Int, pix_center[1] + width), blob[b].H)
        sub_rows_h = h_min:h_max

        w_min = max(floor(Int, (pix_center[2] - width)), 1)
        w_max = min(ceil(Int, pix_center[2] + width), blob[b].W)
        sub_rows_w = w_min:w_max
        tiled_blob[b] = fill(ImageTile(blob[b], sub_rows_h, sub_rows_w), 1, 1)
    end
    tiled_blob
end


function test_blob()
  # A lot of tests are in a single function to avoid having to reload
  # the full image multiple times.
  blob = SDSSIO.load_field_images(RUN, CAMCOL, FIELD, datadir)

  for b=1:length(blob)
    @test !blob[b].constant_background
  end
  fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" datadir RUN CAMCOL FIELD
  cat_entries = SDSSIO.read_photoobj_celeste(fname)

  tiled_blob, mp = initialize_celeste(blob, cat_entries,
                                                patch_radius=1e-6,
                                                fit_psf=false, tile_width=20)

  # Just check some basic facts about the catalog.
  @test length(cat_entries) == 805
  @test 0 < sum([ce.is_star for ce in cat_entries]) < 805

  # Find an object near the middle of the image.
  ctr = WCS.pix_to_world(blob[3].wcs, [blob[3].H / 2, blob[3].W / 2])
  dist = [sqrt((ce.pos[1] - ctr[1])^2 + (ce.pos[2] - ctr[2])^2)
          for ce in cat_entries]
  obj_index = findmin(dist)[2]  # index of closest object
  obj_loc = cat_entries[obj_index].pos  # location of closest object

  # Test cropping.
  width = 5.0
  cropped_blob = crop_blob_to_location(blob, width, obj_loc);
  for b=1:length(blob)
    # Check that it only has one tile of the right size containing the object.
    @assert length(cropped_blob[b]) == 1
    @test 2 * width <= cropped_blob[b][1].h_width <= 2 * (width + 1)
    @test 2 * width <= cropped_blob[b][1].w_width <= 2 * (width + 1)
    patches = vec(mp.patches[:, b])
    tile_sources = Model.get_local_sources(cropped_blob[b][1],
                                           Model.patch_ctrs_pix(patches),
                                           Model.patch_radii_pix(patches))
    @test obj_index in tile_sources
  end

  # Test get_source_psf at point while we have the blob loaded.
  test_b = 3
  img = blob[test_b]
  mp_obj = ModelInit.initialize_model_params(tiled_blob, blob,
                                             cat_entries[obj_index:obj_index])
  pixel_loc = WCS.world_to_pix(img.wcs, obj_loc);
  original_psf_val = img.raw_psf_comp(pixel_loc[1], pixel_loc[2])

  original_psf_celeste = PSF.fit_raw_psf_for_celeste(original_psf_val)[1];
  fit_original_psf_val = PSF.get_psf_at_point(original_psf_celeste);

  obj_psf = PSF.get_source_psf(mp_obj.vp[1][ids.u], img)[1];
  obj_psf_val = PSF.get_psf_at_point(obj_psf);

  # The fits should match exactly.
  @test_approx_eq_eps(obj_psf_val, fit_original_psf_val, 1e-6)

  # The raw psf will not be as good.
  @test_approx_eq_eps(obj_psf_val, original_psf_val, 1e-2)

  mp_several =
    ModelInit.initialize_model_params(
      tiled_blob, blob, [cat_entries[1]; cat_entries[obj_index]]);

  # The second set of vp is the object of interest
  point_patch_psf = PSF.get_psf_at_point(mp_several.patches[2, test_b].psf);
  @test_approx_eq_eps(obj_psf_val, point_patch_psf, 1e-6)
end


function test_fit_object_psfs()
  blob = SDSSIO.load_field_images(RUN, CAMCOL, FIELD, datadir);
  fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" datadir RUN CAMCOL FIELD
  cat_entries = SDSSIO.read_photoobj_celeste(fname);

  # Only test a few catalog entries.
  relevant_sources = collect(3:4)
  test_sources = collect(1:5)

  tiled_blob, mp = initialize_celeste(
    blob, cat_entries[test_sources], fit_psf=false);
  ModelInit.fit_object_psfs!(mp, relevant_sources, blob);

  tiled_blob, mp_psf_init = initialize_celeste(
    blob, cat_entries[test_sources], fit_psf=true);

  for b in 1:length(blob), s in relevant_sources
    @test_approx_eq(
      blob[b].raw_psf_comp(mp.patches[s, b].pixel_center...),
      blob[b].raw_psf_comp(mp_psf_init.patches[s, b].pixel_center...))
  end
end


function test_stamp_get_object_psf()
  stamp_blob, stamp_mp, body = gen_sample_star_dataset();
  img = stamp_blob[3];
  obj_index =  stamp_mp.vp[1][ids.u]
  pixel_loc = WCS.world_to_pix(img.wcs, obj_index)
  original_psf_val = PSF.get_psf_at_point(img.psf);

  obj_psf_val =
    PSF.get_psf_at_point(PSF.get_source_psf(stamp_mp.vp[1][ids.u], img)[1])
  @test_approx_eq_eps(obj_psf_val, original_psf_val, 1e-6)
end


function test_get_tiled_image_source()
  # Test that an object only occurs the appropriate tile's local sources.
  blob, mp, body, tiled_blob = gen_sample_star_dataset();

  mp = ModelInit.initialize_model_params(
    tiled_blob, blob, body; patch_radius=1e-6)

  tiled_img = Model.break_image_into_tiles(blob[3], 10);
  for hh in 1:size(tiled_img)[1], ww in 1:size(tiled_img)[2]
    tile = tiled_img[hh, ww]
    loc = Float64[mean(tile.h_range), mean(tile.w_range)]
    for b = 1:5
      mp.vp[1][ids.u] = loc
      pixel_center = WCSUtils.world_to_pix(blob[b].wcs, loc)
      wcs_jacobian = WCSUtils.pixel_world_jacobian(blob[b].wcs, pixel_center)
      radius_pix = maxabs(eigvals(wcs_jacobian)) * 1e-6
      mp.patches[1, b] = SkyPatch(loc,
                                  radius_pix,
                                  blob[b].psf,
                                  wcs_jacobian,
                                  pixel_center)
    end
    patches = vec(mp.patches[:, 3])
    local_sources = Model.get_tiled_image_sources(tiled_img,
                                                  Model.patch_ctrs_pix(patches),
                                                  Model.patch_radii_pix(patches))
    @test local_sources[hh, ww] == Int[1]
    for hh2 in 1:size(tiled_img)[1], ww2 in 1:size(tiled_img)[2]
      if (hh2 != hh) || (ww2 != ww)
        @test local_sources[hh2, ww2] == Int[]
      end
    end
  end
end


function test_local_source_candidate()
  blob, mp, body, tiled_blob = gen_n_body_dataset(100);

  # This is run by gen_n_body_dataset but put it here for safe testing in
  # case that changes.
  mp = ModelInit.initialize_model_params(tiled_blob, blob, body);

  for b=1:length(tiled_blob)
    # Get the sources by iterating over everything.
    patches = vec(mp.patches[:,b])

    tile_sources = Model.get_tiled_image_sources(tiled_blob[b],
                                                 Model.patch_ctrs_pix(patches),
                                                 Model.patch_radii_pix(patches))

    # Check that all the actual sources are candidates and that this is the
    # same as what is returned by initialize_model_params.
    HH, WW = size(tile_sources)
    for h=1:HH, w=1:WW
      # Get a set of candidates.
      candidates = Model.local_source_candidates(
                        tiled_blob[b][h, w],
                        Model.patch_ctrs_pix(patches),
                        Model.patch_radii_pix(patches))
      @test setdiff(tile_sources[h, w], candidates) == []
      @test tile_sources[h, w] == mp.tile_sources[b][h, w]
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
    tiled_blob, mp =
      initialize_celeste(blob, cat, tile_width=typemax(Int));

    for b=1:length(blob)
      @assert size(tiled_blob[b]) == (1, 1)
      tile_image = ElboDeriv.tile_predicted_image(
        tiled_blob[b][1,1], mp, mp.tile_sources[b][1,1]);

      pixel_center = WCS.world_to_pix(blob[b].wcs, cat[1].pos)
      radius = Model.choose_patch_radius(
        pixel_center, cat[1], blob[b].psf, blob[b])

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


function test_copy_model_params()
  # A lot of tests are in a single function to avoid having to reload
  # the full image multiple times.
  images = SDSSIO.load_field_images(RUN, CAMCOL, FIELD, datadir);

  # Make sure that ModelParams can handle more than five images (issue #203)
  push!(images, deepcopy(images[1]));
  fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" datadir RUN CAMCOL FIELD
  cat_entries = SDSSIO.read_photoobj_celeste(fname);

  tiled_images, mp_all =
    initialize_celeste(images, cat_entries, fit_psf=false, tile_width=20);

  # Pick a single object of interest.
  obj_s = 100
  objid = mp_all.objids[obj_s]
  relevant_sources = ModelInit.get_relevant_sources(mp_all, obj_s);
  mp_all.active_sources = [ obj_s ];
  mp = ModelParams(mp_all, relevant_sources);

  s = findfirst(mp.objids, objid)
  @test s > 0

  # Fit with both and make sure you get the same answer.
  ModelInit.fit_object_psfs!(mp_all, relevant_sources, images);
  ModelInit.fit_object_psfs!(mp, collect(1:mp.S), images);

  @test mp.S == length(relevant_sources)
  for sa in 1:length(relevant_sources)
    s = relevant_sources[sa]
    @test_approx_eq mp.vp[sa] mp_all.vp[s]
  end

  elbo_all = ElboDeriv.elbo(tiled_images, mp_all);
  elbo = ElboDeriv.elbo(tiled_images, mp);

  @test_approx_eq elbo_all.v elbo.v
  @test_approx_eq elbo_all.d elbo.d
  @test_approx_eq elbo_all.h elbo.h
end


test_blob()
test_stamp_get_object_psf()
test_get_tiled_image_source()
test_local_source_candidate()
test_set_patch_size()
test_copy_model_params()
