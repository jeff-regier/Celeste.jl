using Base.Test
using DataFrames

import SloanDigitalSkySurvey.PSF
import WCS

using Celeste: Types, SampleData
import Celeste: ModelInit, SkyImages, ElboDeriv, Synthetic, SDSSIO

println("Running SkyImages tests.")

const RUN = 3900
const CAMCOL = 6
const FIELD = 269

function test_blob()
  # A lot of tests are in a single function to avoid having to reload
  # the full image multiple times.
    run = 3900
    camcol = 6
    field = 269

  blob = SkyImages.read_sdss_field(run, camcol, field, datadir)

  for b=1:5
    @test !blob[b].constant_background
  end
  fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" datadir run camcol field
  cat_entries = SkyImages.read_photoobj_celeste(fname)

  tiled_blob, mp = ModelInit.initialize_celeste(blob, cat_entries,
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
  cropped_blob = SkyImages.crop_blob_to_location(blob, width, obj_loc);
  for b=1:5
    # Check that it only has one tile of the right size containing the object.
    @assert length(cropped_blob[b]) == 1
    @test 2 * width <= cropped_blob[b][1].h_width <= 2 * (width + 1)
    @test 2 * width <= cropped_blob[b][1].w_width <= 2 * (width + 1)
    tile_sources =
      SkyImages.get_local_sources(cropped_blob[b][1], mp.patches[:,b][:])
    @test obj_index in tile_sources
  end

  # Test get_source_psf at point while we have the blob loaded.
  test_b = 3
  img = blob[test_b]
  mp_obj = ModelInit.initialize_model_params(tiled_blob, blob,
                                             cat_entries[obj_index:obj_index])
  pixel_loc = WCS.world_to_pix(img.wcs, obj_loc);
  original_psf_val = img.raw_psf_comp(pixel_loc[1], pixel_loc[2])

  original_psf_celeste = SkyImages.fit_raw_psf_for_celeste(original_psf_val);
  fit_original_psf_val = PSF.get_psf_at_point(original_psf_celeste);

  obj_psf = SkyImages.get_source_psf(mp_obj.vp[1][ids.u], img);
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


function test_stamp_get_object_psf()
  stamp_blob, stamp_mp, body = gen_sample_star_dataset();
  img = stamp_blob[3];
  obj_index =  stamp_mp.vp[1][ids.u]
  pixel_loc = WCS.world_to_pix(img.wcs, obj_index)
  original_psf_val = PSF.get_psf_at_point(img.psf);

  obj_psf_val =
    PSF.get_psf_at_point(SkyImages.get_source_psf(stamp_mp.vp[1][ids.u], img))
  @test_approx_eq_eps(obj_psf_val, original_psf_val, 1e-6)
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
    local_sources = ModelInit.get_tiled_image_sources(img, tiled_img, patches)
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
    patches = mp.patches[:,b][:]
    tile_sources = ModelInit.get_tiled_image_sources(blob[b],
                                tiled_blob[b], patches)

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
                              0.1, .01, pi/4, gal_scale, "sample", 0) ]
  end

  srand(1)
  blob0 = SkyImages.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf");
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
      ModelInit.initialize_celeste(blob, cat, tile_width=typemax(Int));

    for b=1:5
      @assert size(tiled_blob[b]) == (1, 1)
      tile_image = ElboDeriv.tile_predicted_image(
        tiled_blob[b][1,1], mp, mp.tile_sources[b][1,1]);

      pixel_center = WCS.world_to_pix(blob[b].wcs, cat[1].pos)
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


function test_stitch_object_tiles()
  # Just test that these functions run without errors.
  blob, mp, body, tiled_blob = gen_n_body_dataset(100, seed=42);

  image, h_range, w_range =
    SkyImages.stitch_object_tiles(1, 3, mp, tiled_blob);
  @test size(image) == (diff(h_range)[1] + 1, diff(w_range)[1] + 1)

  image, h_range, w_range =
    SkyImages.stitch_object_tiles(1, 3, mp, tiled_blob, predicted = true);
  @test size(image) == (diff(h_range)[1] + 1, diff(w_range)[1] + 1)
end


function test_least_squares_psf()
  # open FITS file containing PSF for each band
  psf_filename =
    @sprintf("%s/psField-%06d-%d-%04d.fit", datadir, RUN, CAMCOL, FIELD)
  psf_fits = FITSIO.FITS(psf_filename);
  raw_psf_comp = SDSSIO.read_psf(psf_fits, band_letters[1]);
  close(psf_fits)

  # psf = PSF.get_psf_at_point(500.0, 500.0, raw_psf_comp);
  psf = raw_psf_comp(500., 500.);

  opt_result, mu_vec, sigma_vec, weight_vec =
    SkyImages.fit_psf_gaussians_least_squares(psf, K=2, ftol=1e-5);

  x_mat = SkyImages.get_x_matrix_from_psf(psf);
  psf_fit = SkyImages.render_psf(opt_result.minimum, x_mat);

  @test_approx_eq sum((psf_fit - psf) .^ 2) opt_result.f_minimum
  @test 0 < opt_result.f_minimum < 1e-3

end


function test_psf_transforms()

  mu_vec = Vector{Float64}[ Float64[1, 2], Float64[-1, -2], Float64[1, -1] ]
  sigma_vec = Array(Matrix{Float64}, 3)
  sigma_vec[1] = Float64[ 1 0.1; 0.1 1]
  sigma_vec[2] = Float64[ 1 0.3; 0.3 2]
  sigma_vec[3] = Float64[ 0.5 0.2; 0.2 0.5]
  weight_vec = Float64[0.4, 0.6, 0.1]

  par = SkyImages.wrap_parameters(mu_vec, sigma_vec, weight_vec)
  mu_vec_test, sigma_vec_test, weight_vec_test = SkyImages.unwrap_parameters(par)

  for k=1:3
    @test_approx_eq mu_vec_test[k] mu_vec[k]
    @test_approx_eq sigma_vec_test[k] sigma_vec[k]
    @test_approx_eq weight_vec_test[k] weight_vec[k]
  end

end


test_least_squares_psf()
test_psf_transforms()
test_blob()
test_stamp_get_object_psf()
test_get_tiled_image_source()
test_local_source_candidate()
test_set_patch_size()
test_stitch_object_tiles()
