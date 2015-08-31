#!/usr/bin/env julia

using Celeste
using CelesteTypes

import Synthetic
import ModelInit

const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

function small_image_profile()
    srand(1)

    blob0 = Images.load_stamp_blob(dat_dir, "164.4311-39.0359");
    for b in 1:5
        blob0[b].H, blob0[b].W = 1000, 1000
    end

    fluxes = [4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]

    S = 100
    locations = rand(S, 2) .* 1000.
    world_locations = WCS.pixel_to_world(blob0[3].wcs, locations)

    S_bodies = CatalogEntry[CatalogEntry(world_locations[s, :][:], true,
        fluxes, fluxes, 0.1, .7, pi/4, 4.) for s in 1:S];

    blob = Synthetic.gen_blob(blob0, S_bodies);
    world_radius_pts = WCS.pixel_to_world(blob[3].wcs, [20. 20.; 0. 0.])
    world_radius = maximum(abs(world_radius_pts[1,:] - world_radius_pts[2,:]))
    mp = ModelInit.cat_init(S_bodies, patch_radius=world_radius, tile_width=50);
    tiled_blob = ModelInit.initialize_celeste!(blob, mp; patch_radius=world_radius);

    ############### Initialize for debugging
    patch_radius = world_radius

    @assert size(mp.patches)[1] == mp.S
    @assert size(mp.patches)[2] == length(blob)

    println("Initiazling Celeste.")
    println("Initializing patches...")
    for s=1:mp.S
      for b = 1:length(blob)
        Images.set_patch_wcs!(mp.patches[s, b], blob[b].wcs)
        mp.patches[s, b].center = mp.vp[s][ids.u]
        mp.patches[s, b].radius = patch_radius
      end
    end
    println("Setting patch psfs...")
    Images.set_patch_psfs!(blob, mp)

    println("Breaking blob into tiles...")
    tiled_blob = Images.break_blob_into_tiles(blob, mp.tile_width);
    @assert length(mp.tile_sources) == length(blob)


    println("Getting sources...")
    for b=1:length(blob)
      println("...for band $b")
      mp.tile_sources[b] = get_tiled_image_sources(tiled_blob[b], blob[b].wcs, mp)
    end
    ################




    tiled_blob = ModelInit.initialize_celeste!(blob, mp);

    println("Calculating ELBO.")
    elbo = ElboDeriv.elbo(tiled_blob, mp)
end


println("Running with ", length(workers()), " processors.")

Profile.init(10^7, 0.001)
small_image_profile()
@profile small_image_profile()
#Profile.print()
Profile.print(format=:flat)
