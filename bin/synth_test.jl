#!/usr/bin/env julia

using Celeste
using CelesteTypes

import Synthetic


function synth_infer_and_cache(stamp_id)
#    srand(3)
    blob0 = SDSS.load_stamp_blob(ENV["STAMP"], stamp_id)

    bright(ce) = sum(ce.star_fluxes) > 3 || sum(ce.gal_fluxes) > 3
    inbounds(ce) = ce.pos[1] > -10. && ce.pos[2] > -10 &&
        ce.pos[1] < 61 && ce.pos[2] < 61

    cat_coadd = SDSS.load_stamp_catalog(ENV["STAMP"], "s82-$stamp_id", blob0)
    cat_coadd = filter(bright, cat_coadd)
    cat_coadd = filter(inbounds, cat_coadd)

    blob = Synthetic.gen_blob(blob0, cat_coadd)

    cat_primary = SDSS.load_stamp_catalog(ENV["STAMP"], stamp_id, blob, match_blob=true)
    cat_primary = filter(bright, cat_primary)
    cat_primary = filter(inbounds, cat_primary)

    mp = ModelInit.cat_init(cat_primary)

	OptimizeElbo.maximize_elbo(blob, mp)

    f = open(ENV["MODEL"]"/S-$stamp_id.dat", "w+")
    serialize(f, mp)
    close(f)
end

if length(ARGS) > 0
	synth_infer_and_cache(ARGS[1])
end
	
