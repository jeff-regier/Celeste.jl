#!/usr/bin/env julia

using DataFrames
import JLD

import Celeste.AccuracyBenchmark
import Celeste.Log
import Celeste.ParallelRun

data_rows = DataFrame[]
for jld_path in ARGS
    Log.info("Reading $jld_path...")
    sources = JLD.load(jld_path)["results"]
    Log.info("Found $(length(sources)) sources")
    for source in sources
        push!(data_rows,
              AccuracyBenchmark.variational_parameters_to_data_frame_row(
                  source.vs)
              )
    end
end
data = vcat(data_rows...)

#################################################

# a hack, having this here.
# probably having Celeste flag bad predictions in the OptimizedSource
# struct is the right way to do this.

import Celeste: Model, SDSSIO

datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
rcf = SDSSIO.RunCamcolField(4263, 5, 119)
strategy = SDSSIO.PlainFITSStrategy(datadir)
images = SDSSIO.load_field_images(strategy, [rcf])
badsky = 0

for row in eachrow(data)
    b = 4
    img = images[b]
    ce = AccuracyBenchmark.make_catalog_entry(row)
    p = Model.SkyPatch(img, ce)
    sp = Model.SkyPatch(img, ce, radius_override_pix=50)

    h = p.bitmap_offset[1] + round(Int, p.radius_pix)
    w = p.bitmap_offset[2] + round(Int, p.radius_pix)
    claimed_sky = img.sky[h, w] * img.nelec_per_nmgy[h]

    H2, W2 = size(sp.active_pixel_bitmap)
    h_range = (sp.bitmap_offset[1] + 1):(sp.bitmap_offset[1] + H2)
    w_range = (sp.bitmap_offset[2] + 1):(sp.bitmap_offset[2] + W2)
    observed_sky = median(filter(!isnan, img.pixels[h_range, w_range]))
    if (claimed_sky + 5) < observed_sky
        row[:flux_r_nmgy] = NaN
        badsky += 1
    end
end
@show badsky

#################################################

if length(ARGS) == 1
    csv_path = string(splitext(ARGS[1])[1], ".csv")
else
    csv_path = "celeste_catalog.csv"
end
Log.info("Writing $csv_path...")
writetable(csv_path, data)
