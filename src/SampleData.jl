
module SampleData

VERSION < v"0.4.0-dev" && using Docile
using Distributions
using CelesteTypes

import SloanDigitalSkySurvey: WCS
import SkyImages

import ModelInit
import Synthetic

export empty_model_params
export dat_dir, sample_ce, perturb_params
export sample_star_fluxes, sample_galaxy_fluxes
export gen_sample_star_dataset, gen_sample_galaxy_dataset
export gen_two_body_dataset, gen_three_body_dataset, gen_n_body_dataset

const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

const sample_star_fluxes = [
    4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]
const sample_galaxy_fluxes = [
    1.377666E+01, 5.635334E+01, 1.258656E+02,
    1.884264E+02, 2.351820E+02] * 100  # 1x wasn't bright enough


function empty_model_params(S::Int)
    vp = [ ModelInit.init_source([ 0., 0. ]) for s in 1:S ]
    ModelParams(vp, ModelInit.sample_prior(), 1)
end


function sample_ce(pos, is_star::Bool)
    CatalogEntry(pos, is_star, sample_star_fluxes, sample_galaxy_fluxes,
        0.1, .7, pi/4, 4.)
end


function perturb_params(mp) # for testing derivatives != 0
    for vs in mp.vp
        vs[ids.a] = [ 0.4, 0.6 ]
        vs[ids.u[1]] += .8
        vs[ids.u[2]] -= .7
        vs[ids.r1] /= 10
        vs[ids.r2] *= 25.
        vs[ids.e_dev] += 0.05
        vs[ids.e_axis] += 0.05
        vs[ids.e_angle] += pi/10
        vs[ids.e_scale] *= 1.2
        vs[ids.c1] += 0.5
        vs[ids.c2] =  1e-1
    end
end


function gen_sample_star_dataset(; perturb=true)
    srand(1)
    blob0 = SkyImages.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23
        blob0[b].wcs = WCS.wcs_id
    end
    one_body = [sample_ce([10.1, 12.2], true),]
    blob = Synthetic.gen_blob(blob0, one_body)
    mp = ModelInit.cat_init(one_body)
    if perturb
        perturb_params(mp)
    end
    tiled_blob = ModelInit.initialize_tiles_and_patches!(blob, mp)
    blob, mp, one_body, tiled_blob
end


function gen_sample_galaxy_dataset(; perturb=true)
    srand(1)
    blob0 = SkyImages.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23
        blob0[b].wcs = WCS.wcs_id
    end
    one_body = [sample_ce([8.5, 9.6], false),]
    blob = Synthetic.gen_blob(blob0, one_body)
    mp = ModelInit.cat_init(one_body)
    if perturb
        perturb_params(mp)
    end
    tiled_blob = ModelInit.initialize_tiles_and_patches!(blob, mp)
    blob, mp, one_body, tiled_blob
end

function gen_two_body_dataset(; perturb=true)
    # A small two-body dataset for quick unit testing.  These objects
    # will be too close to be identifiable.

    srand(1)
    blob0 = SkyImages.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23
        blob0[b].wcs = WCS.wcs_id
    end
    two_bodies = [
        sample_ce([4.5, 3.6], false),
        sample_ce([10.1, 12.1], true)
    ]
    blob = Synthetic.gen_blob(blob0, two_bodies)
    mp = ModelInit.cat_init(two_bodies)
    if perturb
        perturb_params(mp)
    end

    tiled_blob = ModelInit.initialize_tiles_and_patches!(blob, mp)
    blob, mp, two_bodies, tiled_blob
end



function gen_three_body_dataset(; perturb=true)
    srand(1)
    blob0 = SkyImages.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 112, 238
        blob0[b].wcs = WCS.wcs_id
    end
    three_bodies = [
        sample_ce([4.5, 3.6], false),
        sample_ce([60.1, 82.2], true),
        sample_ce([71.3, 100.4], false),
    ]
    blob = Synthetic.gen_blob(blob0, three_bodies)
    mp = ModelInit.cat_init(three_bodies)
    if perturb
        perturb_params(mp)
    end

    tiled_blob = ModelInit.initialize_tiles_and_patches!(blob, mp)
    blob, mp, three_bodies, tiled_blob
end


@doc """
Generate a large dataset with S randomly placed bodies.
""" ->
function gen_n_body_dataset(S::Int64; patch_pixel_radius=20., tile_width=50)
  blob0 = SkyImages.load_stamp_blob(dat_dir, "164.4311-39.0359");
  img_size = 1000
  for b in 1:5
      blob0[b].H, blob0[b].W = img_size, img_size
  end

  fluxes = [4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]

  locations = rand(S, 2) .* float(img_size)
  world_locations = WCS.pixel_to_world(blob0[3].wcs, locations)

  S_bodies = CatalogEntry[CatalogEntry(world_locations[s, :][:], true,
      fluxes, fluxes, 0.1, .7, pi/4, 4.) for s in 1:S];

  blob = Synthetic.gen_blob(blob0, S_bodies);
  world_radius_pts =
    WCS.pixel_to_world(blob[3].wcs,
                       [patch_pixel_radius patch_pixel_radius; 0. 0.])
  world_radius = maximum(abs(world_radius_pts[1,:] - world_radius_pts[2,:]))
  mp = ModelInit.cat_init(S_bodies, tile_width=tile_width);
  tiled_blob = ModelInit.initialize_tiles_and_patches!(
    blob, mp; patch_radius=world_radius);

  blob, mp, S_bodies, tiled_blob
end

end # End module
