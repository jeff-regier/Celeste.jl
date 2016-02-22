using Celeste
using Base.Test
using CelesteTypes
using SampleData
using DataFrames

import ModelInit
import SkyImages
import SloanDigitalSkySurvey: SDSS, WCSUtils, PSF

println("Running SkyImages tests.")

field_dir = joinpath(dat_dir, "sample_field")
run_num = "003900"
camcol_num = "6"
field_num = "0269"

function test_interp_sky()
    data = [1.  2.  3.  4.;
            5.  6.  7.  8.;
            9. 10. 11. 12]
    xcoords = [0.1, 2.5]
    ycoords = [0.5, 2.5, 4.]
    result = SkyImages.interp_sky(data, xcoords, ycoords)
    @test size(result) == (2, 3)
    @test_approx_eq result[1, 1] 1.0
    @test_approx_eq result[2, 1] 7.0
    @test_approx_eq result[1, 2] 2.5
    @test_approx_eq result[2, 2] 8.5
    @test_approx_eq result[1, 3] 4.0
    @test_approx_eq result[2, 3] 10.0
end

function test_blob()
  # A lot of tests are in a single function to avoid having to reload
  # the full image multiple times.

  blob = SkyImages.load_sdss_blob(field_dir, run_num, camcol_num, field_num);
  for b=1:5
    @test !blob[b].constant_background
  end
  cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);
  cat_entries = SkyImages.convert_catalog_to_celeste(cat_df, blob);
  @test ASCIIString[cat_entry.objid for cat_entry in cat_entries ] ==
        convert(Vector{ASCIIString}, cat_df[:objid])
  tiled_blob, mp =
    ModelInit.initialize_celeste(blob, cat_entries, patch_radius=1e-6,
                                 fit_psf=false);

  # Just check some basic facts about the catalog.
  @test size(cat_df)[1] == 805
  @test length(cat_entries) == size(cat_df)[1]
  @test sum([ cat_entry.is_star for cat_entry in cat_entries ]) ==
        sum(cat_df[:is_star])

  # Find an object near the middle of the image.
  #img_center = Float64[ median(cat_df[:ra]), median(cat_df[:dec]) ]
  img_center =
    WCSUtils.pix_to_world(blob[3].wcs, Float64[blob[3].H / 2, blob[3].W / 2])
  dist = by(cat_df, :objid,
     df -> DataFrame(dist=(df[:ra] - img_center[1]).^2 +
                          (df[:dec] - img_center[2]).^2))
  obj_rows = dist[:dist] .== minimum(dist[:dist])
  @assert sum(obj_rows) == 1
  obj_loc = Float64[ cat_df[obj_rows, :ra][1], cat_df[obj_rows, :dec][1]]
  objid = cat_df[obj_rows, :objid][1]
  obj_row = find(obj_rows)[1]

  # # Test cropping.
  width = 5.0
  cropped_blob = SkyImages.crop_blob_to_location(blob, width, obj_loc);
  for b=1:5
    # Check that it only has one tile of the right size containing the object.
    @assert length(cropped_blob[b]) == 1
    @test 2 * width <= cropped_blob[b][1].h_width <= 2 * (width + 1)
    @test 2 * width <= cropped_blob[b][1].w_width <= 2 * (width + 1)
    tile_sources =
      SkyImages.get_local_sources(cropped_blob[b][1], mp.patches[:,b][:])
    @test obj_row in tile_sources
  end

  # Test get_source_psf at point while we have the blob loaded.
  test_b = 3
  img = blob[test_b];
  obj_index = find(obj_rows)
  mp_obj = ModelInit.initialize_model_params(
    tiled_blob, blob, cat_entries[obj_index]);
  pixel_loc = WCSUtils.world_to_pix(img.wcs, obj_loc);
  original_psf_val =
    PSF.get_psf_at_point(pixel_loc[1], pixel_loc[2], img.raw_psf_comp);
  original_psf_gmm, scale = PSF.fit_psf_gaussians(original_psf_val);
  original_psf_celeste = SkyImages.convert_gmm_to_celeste(original_psf_gmm, scale);
  fit_original_psf_val = PSF.get_psf_at_point(original_psf_celeste);

  obj_psf = SkyImages.get_source_psf(mp_obj.vp[1][ids.u], img);
  obj_psf_val = PSF.get_psf_at_point(obj_psf);

  # The fits should match exactly.
  @test_approx_eq_eps(obj_psf_val[:], fit_original_psf_val[:], 1e-6)

  # The raw psf will not be as good.
  @test_approx_eq_eps(obj_psf_val[:], original_psf_val[:], 1e-2)

  mp_several =
    ModelInit.initialize_model_params(
      tiled_blob, blob, [cat_entries[1]; cat_entries[obj_index]]);

  # The second set of vp is the object of interest
  point_patch_psf = PSF.get_psf_at_point(mp_several.patches[2, test_b].psf);
  @test_approx_eq_eps(obj_psf_val[:], point_patch_psf[:], 1e-6)
end


function test_stamp_get_object_psf()
  stamp_blob, stamp_mp, body = gen_sample_star_dataset();
  img = stamp_blob[3];
  obj_loc =  stamp_mp.vp[1][ids.u]
  pixel_loc = WCSUtils.world_to_pix(img.wcs, obj_loc)
  original_psf_val = PSF.get_psf_at_point(img.psf);

  obj_psf_val =
    PSF.get_psf_at_point(SkyImages.get_source_psf(stamp_mp.vp[1][ids.u], img))
  @test_approx_eq_eps(obj_psf_val[:], original_psf_val[:], 1e-6)
end


function test_get_tiled_image_source()
  # Test that an object only occurs the appropriate tile's local sources.
  blob, mp, body, tiled_blob = gen_sample_star_dataset();
  img = blob[3];

  mp = ModelInit.initialize_model_params(
    tiled_blob, blob, body; patch_radius=1e-6)

  tiled_img = SkyImages.break_image_into_tiles(img, 10);
  for hh in 1:size(tiled_img)[1], ww in 1:size(tiled_img)[2]
    tile = tiled_img[hh, ww]
    loc = Float64[mean(tile.h_range), mean(tile.w_range)]
    for b = 1:5
      mp.vp[1][ids.u] = loc
      mp.patches[1, b] = SkyPatch(loc, 1e-6, blob[b], fit_psf=false)
    end
    patches = mp.patches[:, 3][:]
    local_sources =
      ModelInit.get_tiled_image_sources(tiled_img, img.wcs, patches)
    @test local_sources[hh, ww] == Int64[1]
    for hh2 in 1:size(tiled_img)[1], ww2 in 1:size(tiled_img)[2]
      if (hh2 != hh) || (ww2 != ww)
        @test local_sources[hh2, ww2] == Int64[]
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
    patches = mp.patches[:,b][:]
    tile_sources =
      ModelInit.get_tiled_image_sources(tiled_blob[b], blob[b].wcs, patches)

    # Get a set of candidates.
    candidates = SkyImages.local_source_candidates(tiled_blob[b], patches);

    # Check that all the actual sources are candidates and that this is the
    # same as what is returned by initialize_model_params.
    @test size(candidates) == size(tile_sources)
    for h=1:size(candidates)[1], w=1:size(candidates)[2]
      @test setdiff(tile_sources[h, w], candidates[h, w]) == []
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
                              0.1, .01, pi/4, gal_scale, "sample") ]
  end

  srand(1)
  blob0 = SkyImages.load_stamp_blob(dat_dir, "164.4311-39.0359");
  img_size = 150
  for b in 1:5
      blob0[b].H, blob0[b].W = img_size, img_size
  end
  fluxes = [4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]

  world_location = WCSUtils.pix_to_world(blob0[3].wcs,
                                         Float64[img_size / 2, img_size / 2])

  for gal_scale in [1.0, 10.0], flux_scale in [0.1, 10.0]
    cat = gal_catalog_from_scale(gal_scale, flux_scale);
    blob = Synthetic.gen_blob(blob0, cat);
    tiled_blob, mp =
      ModelInit.initialize_celeste(blob, cat, tile_width=typemax(Int64));

    for b=1:5
      @assert size(tiled_blob[b]) == (1, 1)
      tile_image = ElboDeriv.tile_predicted_image(
        tiled_blob[b][1,1], mp, mp.tile_sources[b][1,1]);

      pixel_center = WCSUtils.world_to_pix(blob[b].wcs, cat[1].pos)
      radius = ModelInit.choose_patch_radius(
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


function test_get_local_sources()
  world_radius = 2.0
  wcs_jacobian = Float64[0.5 0.1; 0.2 0.6]
  pixel_center = Float64[0, 0]
  world_center = Float64[0, 0]
  patch1 = SkyPatch(world_center, world_radius,
                    PsfComponent[], wcs_jacobian, pixel_center);

  h_width = 10
  w_width = 20
  tile = ImageTile(1, 1, 1, 1:h_width, 1:w_width, h_width, w_width,
                   rand(h_width, w_width), true, 0.5, Array(Float64, 0, 0),
                   0.5, Array(Float64, 0));
  SkyImages.get_local_sources(tile, [ patch ])
end


test_interp_sky()
test_blob()
test_stamp_get_object_psf()
test_get_tiled_image_source()
test_local_source_candidate()
test_set_patch_size()
