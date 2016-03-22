using Base.Test

using Celeste: Types, SampleData
import Celeste: SkyImages, Util, ModelInit, Synthetic
import Celeste.ModelInit: patch_ctrs_pix, patch_radii_pix

println("Running misc tests.")

function test_tile_image()
  blob, mp, three_bodies = gen_three_body_dataset();
  img = blob[3];

  # First with constant background
  tile_width = 20;
  tile = ImageTile(1, 1, img, tile_width);

  tiles = SkyImages.break_image_into_tiles(img, tile_width);
  @test size(tiles) ==
    (round(Int, ceil(img.H  / tile_width)),
     round(Int, ceil(img.W / tile_width)))
  for tile in tiles
    @test tile.b == img.b
    @test tile.pixels == img.pixels[tile.h_range, tile.w_range]
    @test tile.epsilon == img.epsilon
    @test tile.iota == img.iota
    @test tile.constant_background == img.constant_background
  end

  # Then with varying background
  img.constant_background = false
  img.epsilon_mat = rand(size(img.pixels));
  img.iota_vec = rand(size(img.pixels)[1]);
  tiles = SkyImages.break_image_into_tiles(img, tile_width);
  @test size(tiles) == (
    ceil(Int, img.H  / tile_width),
    ceil(Int, img.W / tile_width))
  for tile in tiles
    @test tile.b == img.b
    @test tile.pixels == img.pixels[tile.h_range, tile.w_range]
    @test tile.epsilon_mat == img.epsilon_mat[tile.h_range, tile.w_range]
    @test tile.iota_vec == img.iota_vec[tile.h_range]
    @test tile.constant_background == img.constant_background
  end

  tile = tiles[2, 2]
  for h in 1:tile.h_width, w in 1:tile.w_width
    @test tile.pixels[h, w] == img.pixels[tile.h_range[h], tile.w_range[w]]
  end
end

function test_local_sources()
    # Coarse test that local_sources gets the right objects.

    srand(1)
    blob0 = SkyImages.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf");
    for b in 1:5
        blob0[b].H, blob0[b].W = 112, 238
        blob0[b].wcs = WCSUtils.wcs_id
    end

    three_bodies = [
        sample_ce([4.5, 3.6], false),
        sample_ce([60.1, 82.2], true),
        sample_ce([71.3, 100.4], false),
    ]

    blob = Synthetic.gen_blob(blob0, three_bodies);

    tile = ImageTile(1, 1, blob[3], 1000);
    mp = ModelInit.initialize_model_params(
      fill(fill(tile, 1, 1), 5), blob, three_bodies; patch_radius=20.);
    @test mp.S == 3

    patches = vec(mp.patches[:, 3])
    subset1000 = ModelInit.get_local_sources(tile, patch_ctrs_pix(patches),
                                             patch_radii_pix(patches))
    @test subset1000 == [1,2,3]

    tile_width = 10
    tile = ImageTile(1, 1, blob[3], tile_width);
    ModelInit.initialize_model_params(
      fill(fill(tile, 1, 1), 5), blob, three_bodies; patch_radius=20.);

    patches = vec(mp.patches[:, 3])
    subset10 = ModelInit.get_local_sources(tile, patch_ctrs_pix(patches),
                                           patch_radii_pix(patches))
    @test subset10 == [1]

    last_tile = ImageTile(11, 24, blob[3], tile_width)
    ModelInit.initialize_model_params(
      fill(fill(last_tile, 1, 1), 5), blob, three_bodies; patch_radius=20.)

    patches = vec(mp.patches[:, 3])
    last_subset = ModelInit.get_local_sources(last_tile,
                                              patch_ctrs_pix(patches),
                                              patch_radii_pix(patches))
    @test length(last_subset) == 0

    pop_tile = ImageTile(7, 9, blob[3], tile_width)
    ModelInit.initialize_model_params(
      fill(fill(pop_tile, 1, 1), 5), blob, three_bodies; patch_radius=20.);

    patches = vec(mp.patches[:, 3])
    pop_subset = ModelInit.get_local_sources(pop_tile, patch_ctrs_pix(patches),
                                             patch_radii_pix(patches))

    @test pop_subset == [2,3]
end


function test_local_sources_2()
    # Check that a larger blob gets the same number of objects
    # as a smaller blob.  (This is useful to check edge cases of
    # the polygon logic.)

    srand(1)
    blob0 = SkyImages.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf");
    one_body = [sample_ce([50., 50.], true),]
    for b in 1:5 blob0[b].wcs = WCSUtils.wcs_id end

    for b in 1:5 blob0[b].H, blob0[b].W = 100, 100 end
    small_blob = Synthetic.gen_blob(blob0, one_body);

    for b in 1:5 blob0[b].H, blob0[b].W = 400, 400 end
    big_blob = Synthetic.gen_blob(blob0, one_body);

    small_tiled_blob, mp_small = ModelInit.initialize_celeste(
      small_blob, one_body, patch_radius=35.);
    small_source_tiles =
      [ sum([ length(s) > 0 for s in source ]) for source in mp_small.tile_sources ]

    big_tiled_blob, mp_big = ModelInit.initialize_celeste(
      big_blob, one_body, patch_radius=35.);
    big_source_tiles =
      [ sum([ length(s) > 0 for s in source ]) for source in mp_big.tile_sources ]

    @test all(big_source_tiles .== small_source_tiles)
end


function test_local_sources_3()
    # Test local_sources using world coordinates.

    srand(1)
    test_b = 3 # Will test using this band only
    pix_loc = Float64[50., 50.]
    blob0 = SkyImages.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf");
    body_loc = WCSUtils.pix_to_world(blob0[test_b].wcs, pix_loc)
    one_body = [sample_ce(body_loc, true),]

    # Get synthetic blobs but with the original world coordinates.
    for b in 1:5 blob0[b].H, blob0[b].W = 100, 100 end
    blob = Synthetic.gen_blob(blob0, one_body);
    for b in 1:5 blob[b].wcs = blob0[b].wcs end

    tile_width = 1
    patch_radius_pix = 5.

    # Get a patch radius in world coordinates by looking at the world diagonals of
    # a pixel square of a certain size.
    pix_quad = [0. 0.               patch_radius_pix patch_radius_pix;
                0. patch_radius_pix 0.               patch_radius_pix]
    world_quad = WCSUtils.pix_to_world(blob[test_b].wcs, pix_quad)
    diags = [world_quad[:, i] - world_quad[:, i+2] for i=1:2]
    patch_radius = maximum([sqrt(dot(d, d)) for d in diags])

    tiled_blob, mp = ModelInit.initialize_celeste(
      blob, one_body, patch_radius=patch_radius);

    # Source should be present
    tile = ImageTile(
        round(Int, pix_loc[1] / tile_width),
        round(Int, pix_loc[2] / tile_width),
        blob[test_b],
        tile_width);

    patches = vec(mp.patches[:,test_b])
    @test ModelInit.get_local_sources(tile, patch_ctrs_pix(patches),
                                      patch_radii_pix(patches)) == [1]

    # Source should not match when you're 1 tile and a half away along the diagonal plus
    # the pixel radius from the center of the tile.
    tile = ImageTile(
        ceil(Int, (pix_loc[1] + 1.5 * tile_width * sqrt(2) +
                patch_radius_pix) / tile_width),
        round(Int, pix_loc[2] / tile_width),
        blob[test_b],
        tile_width)
    @test ModelInit.get_local_sources(tile, patch_ctrs_pix(patches),
                                      patch_radii_pix(patches)) == []

    tile = ImageTile(
        round(Int, (pix_loc[1]) / tile_width),
        ceil(Int, (pix_loc[2]  + 1.5 * tile_width * sqrt(2) +
                           patch_radius_pix) / tile_width),
        blob[test_b],
        tile_width)
    @test ModelInit.get_local_sources(tile, patch_ctrs_pix(patches),
                                      patch_radii_pix(patches)) == []

end


function test_tiling()
    srand(1)
    blob0 =SkyImages.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf")
    for b in 1:5
        blob0[b].H, blob0[b].W = 112, 238
    end
    three_bodies = [
        sample_ce([4.5, 3.6], false),
        sample_ce([60.1, 82.2], true),
        sample_ce([71.3, 100.4], false),
    ]
   blob = Synthetic.gen_blob(blob0, three_bodies)

    mp = ModelInit.cat_init(three_bodies)
    elbo = ElboDeriv.elbo(blob, mp)

    mp2 = ModelInit.cat_init(three_bodies, tile_width=10)
    elbo_tiles = ElboDeriv.elbo(blob, mp2)
    @test_approx_eq_eps elbo_tiles.v[1] elbo.v[1] 1e-5

    mp3 = ModelInit.cat_init(three_bodies, patch_radius=30.)
    elbo_patches = ElboDeriv.elbo(blob, mp3)
    @test_approx_eq_eps elbo_patches.v[1] elbo.v[1] 1e-5

    for s in 1:mp.S
        for i in 1:length(1:length(CanonicalParams))
            @test_approx_eq_eps elbo_tiles.d[i, s] elbo.d[i, s] 1e-5
            @test_approx_eq_eps elbo_patches.d[i, s] elbo.d[i, s] 1e-5
        end
    end

    mp4 = ModelInit.cat_init(three_bodies, patch_radius=35., tile_width=10)
    elbo_both = ElboDeriv.elbo(blob, mp4)
    @test_approx_eq_eps elbo_both.v[1] elbo.v[1] 1e-1

    for s in 1:mp.S
        for i in 1:length(1:length(CanonicalParams))
            @test_approx_eq_eps elbo_both.d[i, s] elbo.d[i, s] 1e-1
        end
    end
end


function test_sky_noise_estimates()
    blobs = Array(Blob, 2)
    blobs[1], mp, three_bodies = gen_three_body_dataset()  # synthetic
    blobs[2] = SkyImages.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf")  # real

    for blob in blobs
        for b in 1:5
            sdss_sky_estimate = blob[b].epsilon * blob[b].iota
            crude_estimate = median(blob[b].pixels)
            @test_approx_eq_eps sdss_sky_estimate / crude_estimate 1. .3
        end
    end
end


function test_util_bvn_cov()
    e_axis = .7
    e_angle = pi/5
    e_scale = 2.

    manual_11 = e_scale^2 * (1 + (e_axis^2 - 1) * (sin(e_angle))^2)
    util_11 = Util.get_bvn_cov(e_axis, e_angle, e_scale)[1,1]
    @test_approx_eq util_11 manual_11

    manual_12 = e_scale^2 * (1 - e_axis^2) * (cos(e_angle)sin(e_angle))
    util_12 = Util.get_bvn_cov(e_axis, e_angle, e_scale)[1,2]
    @test_approx_eq util_12 manual_12

    manual_22 = e_scale^2 * (1 + (e_axis^2 - 1) * (cos(e_angle))^2)
    util_22 = Util.get_bvn_cov(e_axis, e_angle, e_scale)[2,2]
    @test_approx_eq util_22 manual_22
end


function test_get_relevant_sources()
  blob, mp, body, tiled_blob = gen_n_body_dataset(100; seed=42);
  mp = ModelInit.initialize_model_params(tiled_blob, blob, body);

  target_s = 1
  relevant_sources = ModelInit.get_relevant_sources(mp, target_s);
  @test length(relevant_sources) > 1 # Just to make sure the test is valid

  @test target_s in relevant_sources
  for b in 1:length(mp.tile_sources), tile_sources in mp.tile_sources[b]
    if target_s in tile_sources
      @test all(Bool[ s in relevant_sources for s in tile_sources ])
    end
  end
end

####################################################

test_tile_image()
test_util_bvn_cov()
test_sky_noise_estimates()
test_local_sources()
test_local_sources_2()
test_local_sources_3()
test_get_relevant_sources()
