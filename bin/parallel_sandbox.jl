using Celeste
using CelesteTypes

using DataFrames
using SampleData

import Images
import ElboDeriv

VERSION < v"0.4.0-dev" && using Docile

##################
# Load a stamp to check out the psf and wcs

blob, mp, three_bodies = gen_three_body_dataset();
img = blob[3];

##############
# Setup

NumType = Float64

@assert length(mp.patches) == mp.S
for s=1:mp.S
    ElboDeriv.set_patch_wcs!(mp.patches[s], img.wcs)
    # TODO: Also set a local psf here.
end

star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(img.psf, mp, 3);
sbs = [ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S];

mp.tile_width = 20
WW = int(ceil(img.W / mp.tile_width))
HH = int(ceil(img.H / mp.tile_width))
tiles = ImageTile[ ImageTile(hh, ww, img) for ww=1:WW, hh=1:HH];


###############
# Serial

accum = zero_sensitive_float(CanonicalParams, NumType, mp.S);

i = 1
for tile in tiles
    println("tile $i of $(length(tiles))"); i += 1;
    ElboDeriv.tile_likelihood!(tile, mp, sbs, star_mcs, gal_mcs, accum);
end

accum.v += -sum(lfact(img.pixels[!isnan(img.pixels)]));

###############
# Parallel


function tile_likelihood(tile, mp, sbs, star_mcs, gal_mcs)
  tile_sources = local_sources(tile, mp)
  h_range, w_range = tile_range(tile, mp.tile_width)
  accum = zero_sensitive_float(CanonicalParams, NumType, mp.S);
  ElboDeriv.tile_likelihood!(tile, mp, sbs, star_mcs, gal_mcs, accum);
  accum
end


function +{NumType <: Number}(
    sf1::SensitiveFloat{CanonicalParams, NumType},
    sf2::SensitiveFloat{CanonicalParams, NumType})

  S = size(sf1.d)[2]
  @assert size(sf1.d) == size(sf2.d)
  sfout = zero_sensitive_float(CanonicalParams, NumType, S);
  sfout.v = sf1.v + sf2.v
  sfout.d = sf1.d + sf2.d
  sfout
end


accum_parallel = @parallel (+) for tile in tiles
  tile_likelihood(tile, mp, sbs, star_mcs, gal_mcs)
end

accum.v += -sum(lfact(img.pixels[!isnan(img.pixels)]));












##############
