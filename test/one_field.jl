#!/usr/bin/env julia

using Celeste
using FITSIO
using CelesteTypes


field_dir = "/home/jeff/Dropbox/astronomy/dat/sample_field"
run_num = "003900"
camcol_num = "6"
frame_num = "0269"

cat = SDSS.load_catalog(field_dir, run_num, camcol_num, frame_num)
mp = ModelInit.cat_init(cat)
println(mp.S)
blob = SDSS.load_field(field_dir, run_num, camcol_num, frame_num)
println(ElboDeriv.elbo(blob, mp), patch=20., tile=10)

