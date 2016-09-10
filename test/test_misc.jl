using Base.Test

import Celeste.Model: patch_ctrs_pix, patch_radii_pix


function test_tile_image()
    blob, ea, three_bodies = gen_three_body_dataset()
    img = blob[3]
    tile_width = 20
    img.epsilon_mat = rand(size(img.pixels))
    img.iota_vec = rand(size(img.pixels, 1))
    tiles = Model.TiledImage(img; tile_width=tile_width).tiles
    @test size(tiles) == (
        ceil(Int, img.H    / tile_width),
        ceil(Int, img.W / tile_width))
    for tile in tiles
        @test tile.b == img.b
        @test tile.pixels == img.pixels[tile.h_range, tile.w_range]
        @test tile.epsilon_mat[2,3] == img.epsilon_mat[tile.h_range, tile.w_range][2,3]
        @test tile.iota_vec[3] == img.iota_vec[tile.h_range][3]
    end

    tile = tiles[2, 2]
    h_width, w_width = size(tile.pixels)
    for h in 1:h_width, w in 1:w_width
        @test tile.pixels[h, w] == img.pixels[tile.h_range[h], tile.w_range[w]]
    end
end

function test_local_sources()
    # Coarse test that local_sources gets the right objects.

    srand(1)
    blob0 = SampleData.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf")
    for b in 1:5
        blob0[b].H, blob0[b].W = 112, 238
        blob0[b].wcs = SampleData.wcs_id
    end

    three_bodies = [
        sample_ce([4.5, 3.6], false),
        sample_ce([60.1, 82.2], true),
        sample_ce([71.3, 100.4], false),
    ]

    blob = Synthetic.gen_blob(blob0, three_bodies)
    tiled_images = TiledImage[TiledImage(img; tile_width=20) for img in blob]

    tile = ImageTile(1, 1, blob[3], 1000)

    ea = make_elbo_args(tiled_images, three_bodies; patch_radius=20.)
    @test ea.S == 3

    patches = vec(ea.patches[:, 3])
    subset1000 = Model.get_local_sources(tile, patch_ctrs_pix(patches),
                                             patch_radii_pix(patches))
    @test subset1000 == [1,2,3]

    tile_width = 10
    tile = ImageTile(1, 1, blob[3], tile_width)
    make_elbo_args(tiled_images, three_bodies; patch_radius=20.)

    patches = vec(ea.patches[:, 3])
    subset10 = Model.get_local_sources(tile, patch_ctrs_pix(patches),
                                           patch_radii_pix(patches))
    @test subset10 == [1]

    last_tile = ImageTile(11, 24, blob[3], tile_width)
    make_elbo_args(tiled_images, three_bodies; patch_radius=20.)

    patches = vec(ea.patches[:, 3])
    last_subset = Model.get_local_sources(last_tile,
                                              patch_ctrs_pix(patches),
                                              patch_radii_pix(patches))
    @test length(last_subset) == 0

    pop_tile = ImageTile(7, 9, blob[3], tile_width)
    make_elbo_args(tiled_images, three_bodies; patch_radius=20.)

    patches = vec(ea.patches[:, 3])
    pop_subset = Model.get_local_sources(pop_tile, patch_ctrs_pix(patches),
                                             patch_radii_pix(patches))

    @test pop_subset == [2,3]
end


function test_local_sources_2()
    # Check that a larger blob gets the same number of objects
    # as a smaller blob.  (This is useful to check edge cases of
    # the polygon logic.)

    srand(1)
    blob0 = SampleData.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf")
    one_body = [sample_ce([50., 50.], true),]
    for b in 1:5 blob0[b].wcs = SampleData.wcs_id end

    for b in 1:5 blob0[b].H, blob0[b].W = 100, 100 end
    small_blob = Synthetic.gen_blob(blob0, one_body)

    for b in 1:5 blob0[b].H, blob0[b].W = 400, 400 end
    big_blob = Synthetic.gen_blob(blob0, one_body)

    ea_small = make_elbo_args(small_blob, one_body, patch_radius=35.)
    small_source_tiles =
      [ sum([ length(s) > 0 for s in source ]) for source in ea_small.tile_source_map ]

    ea_big = make_elbo_args(big_blob, one_body, patch_radius=35.)
    big_source_tiles =
      [ sum([ length(s) > 0 for s in source ]) for source in ea_big.tile_source_map ]

    @test all(big_source_tiles .== small_source_tiles)
end


function test_local_sources_3()
    # Test local_sources using world coordinates.

    srand(1)
    test_b = 3 # Will test using this band only
    pix_loc = Float64[50., 50.]
    blob0 = SampleData.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf")
    body_loc = WCS.pix_to_world(blob0[test_b].wcs, pix_loc)
    one_body = [sample_ce(body_loc, true),]

    # Get synthetic blobs but with the original world coordinates.
    for b in 1:5 blob0[b].H, blob0[b].W = 100, 100 end
    blob = Synthetic.gen_blob(blob0, one_body)
    for b in 1:5 blob[b].wcs = blob0[b].wcs end

    tile_width = 1
    patch_radius_pix = 5.

    # Get a patch radius in world coordinates by looking at the world diagonals of
    # a pixel square of a certain size.
    pix_quad = [0. 0.               patch_radius_pix patch_radius_pix;
                0. patch_radius_pix 0.               patch_radius_pix]
    world_quad = WCS.pix_to_world(blob[test_b].wcs, pix_quad)
    diags = [world_quad[:, i] - world_quad[:, i+2] for i=1:2]
    patch_radius = maximum([sqrt(dot(d, d)) for d in diags])

    ea = make_elbo_args(blob, one_body, patch_radius=patch_radius)

    # Source should be present
    tile = ImageTile(
        round(Int, pix_loc[1] / tile_width),
        round(Int, pix_loc[2] / tile_width),
        blob[test_b],
        tile_width)

    patches = vec(ea.patches[:,test_b])
    @test Model.get_local_sources(tile, patch_ctrs_pix(patches),
                                      patch_radii_pix(patches)) == [1]

    # Source should not match when you're 1 tile and a half away along the diagonal plus
    # the pixel radius from the center of the tile.
    tile = ImageTile(
        ceil(Int, (pix_loc[1] + 1.5 * tile_width * sqrt(2) +
                patch_radius_pix) / tile_width),
        round(Int, pix_loc[2] / tile_width),
        blob[test_b],
        tile_width)
    @test Model.get_local_sources(tile, patch_ctrs_pix(patches),
                                      patch_radii_pix(patches)) == []

    tile = ImageTile(
        round(Int, (pix_loc[1]) / tile_width),
        ceil(Int, (pix_loc[2]  + 1.5 * tile_width * sqrt(2) +
                           patch_radius_pix) / tile_width),
        blob[test_b],
        tile_width)
    @test Model.get_local_sources(tile, patch_ctrs_pix(patches),
                                      patch_radii_pix(patches)) == []

end


function test_sky_noise_estimates()
    blobs = Array(Vector{Image}, 2)
    blobs[1], ea, three_bodies = gen_three_body_dataset()  # synthetic
    blobs[2] = SampleData.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf")  # real

    for blob in blobs
        for b in 1:5
            sdss_sky_estimate = median(blob[b].epsilon_mat) * median(blob[b].iota_vec)
            crude_estimate = median(blob[b].pixels)
            @test_approx_eq_eps sdss_sky_estimate / crude_estimate 1. .3
        end
    end
end


####################################################

test_tile_image()
test_sky_noise_estimates()
test_local_sources()
test_local_sources_2()
test_local_sources_3()
