#!/usr/bin/env julia

# Computationally intensive.  Caches VB optimum in $STAMP.

using Celeste
using CelesteTypes
import Synthetic


function peak_infer_and_cache(stamp_id)
	blob = SDSS.load_stamp_blob(ENV["STAMP"], stamp_id);
	mp = ModelInit.peak_init(blob);

	OptimizeElbo.maximize_elbo(blob, mp)

	f = open(ENV["MODEL"]"/V-$stamp_id.dat", "w+")
	serialize(f, mp)
	close(f)
end


function cat_infer_and_cache(stamp_id)
	blob = SDSS.load_stamp_blob(ENV["STAMP"], stamp_id);
	cat_entries = SDSS.load_stamp_catalog(ENV["STAMP"], stamp_id, blob, match_blob=true)
	mp = ModelInit.cat_init(cat_entries)

	OptimizeElbo.maximize_elbo(blob, mp)

	f = open(ENV["MODEL"]"/V-$stamp_id.dat", "w+")
	serialize(f, mp)
	close(f)
end


function synth_infer_and_cache(stamp_id)
    bright(ce) = sum(ce.star_fluxes) > 3 || sum(ce.gal_fluxes) > 3
    inbounds(ce) = ce.pos[1] > -10. && ce.pos[2] > -10 &&
        ce.pos[1] < 61 && ce.pos[2] < 61

    f = open(ENV["STAMP"]"/cat-synth-$stamp_id.dat")
    cat_synth = deserialize(f)
    close(f)

    blob0 = SDSS.load_stamp_blob(ENV["STAMP"], stamp_id)
    blob = Synthetic.gen_blob(blob0, cat_synth)

    cat_synth = filter(bright, cat_synth)
    cat_synth = filter(inbounds, cat_synth)

    cat_primary = SDSS.load_stamp_catalog(ENV["STAMP"], stamp_id, blob, match_blob=true)
    cat_primary = filter(bright, cat_primary)
    cat_primary = filter(inbounds, cat_primary)

    mp = ModelInit.cat_init(cat_primary)

    OptimizeElbo.maximize_elbo(blob, mp)

    f = open(ENV["MODEL"]"/S-$stamp_id.dat", "w+")
    serialize(f, mp)
    close(f)
end


if length(ARGS) == 2
    if ARGS[1] == "V"
        cat_infer_and_cache(ARGS[2])
    elseif ARGS[1] == "S"
        synth_infer_and_cache(ARGS[2])
    end
end

