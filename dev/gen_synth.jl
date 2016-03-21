#!/usr/bin/env julia

using Celeste
using CelesteTypes

import Synthetic
import SloanDigitalSkySurvey: SDSS

function gen_synth_cat(stamp_id)
    blob0 = Images.load_stamp_blob(ENV["STAMP"], stamp_id)
    cat_coadd = SDSS.load_stamp_catalog(ENV["STAMP"], "s82-$stamp_id", blob0)
    cat_synth = Synthetic.synthetic_bodies(cat_coadd)

    f = open(ENV["STAMP"]"/cat-synth-$stamp_id.dat", "w+")
    serialize(f, cat_synth)
    close(f)
end

if length(ARGS) >= 0
    f = open(ARGS[1])
    stamp_ids = [strip(line) for line in readlines(f)]
    close(f)

    for stamp_id in stamp_ids
        println(stamp_id)
        gen_synth_cat(stamp_id)
    end
end
