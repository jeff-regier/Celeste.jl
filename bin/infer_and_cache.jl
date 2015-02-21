#!/usr/bin/env julia

# Computationally intensive.  Caches VB optimum in $STAMP.

using Celeste
using CelesteTypes


function peak_infer_and_cache(stamp_id)
	blob = SDSS.load_stamp_blob(ENV["STAMP"], stamp_id);
	mp = ModelInit.peak_init(blob);

	OptimizeElbo.maximize_elbo(blob, mp)

	f = open(ENV["STAMP"]"/V-$stamp_id.dat", "w+")
	serialize(f, mp)
	close(f)
end


function cat_infer_and_cache(stamp_id)
	blob = SDSS.load_stamp_blob(ENV["STAMP"], stamp_id);
	cat_entries = SDSS.load_stamp_catalog(ENV["STAMP"], stamp_id, blob, match_blob=true)
	mp = ModelInit.cat_init(cat_entries)

	OptimizeElbo.maximize_elbo(blob, mp)

	f = open(ENV["STAMP"]"/V-$stamp_id.dat", "w+")
	serialize(f, mp)
	close(f)
end


if length(ARGS) > 0
	cat_infer_and_cache(ARGS[1])
end

