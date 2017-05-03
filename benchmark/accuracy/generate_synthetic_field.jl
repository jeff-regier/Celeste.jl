#!/usr/bin/env julia

using DataFrames

import Celeste.AccuracyBenchmark
import Celeste.ArgumentParse
import Celeste.Model
import Celeste.Synthetic

const PSF_SIGMA_PX = 2.29 # similar to SDSS
const BAND_SKY_LEVEL_NMGY = [0.2696, 0.3425, 0.7748, 1.6903, 4.9176]
const BAND_NELEC_PER_NMGY = [146.9, 838.1, 829.8, 597.2, 129.8]
const OUTPUT_DIRECTORY = joinpath(splitdir(Base.source_path())[1], "output")

parser = Celeste.ArgumentParse.ArgumentParser()
ArgumentParse.add_argument(
    parser,
    "catalog_csv",
    help="Ground truth CSV catalog",
)
parsed_args = ArgumentParse.parse_args(parser, ARGS)

srand(12345)

catalog_data = AccuracyBenchmark.read_catalog(parsed_args["catalog_csv"])
catalog_entries = [
    AccuracyBenchmark.make_catalog_entry(row)
    for row in eachrow(catalog_data)
]
template_images = AccuracyBenchmark.make_template_images(
    catalog_data,
    PSF_SIGMA_PX,
    BAND_SKY_LEVEL_NMGY,
    BAND_NELEC_PER_NMGY,
)
generated_images = Synthetic.gen_blob(template_images, catalog_entries)

catalog_label = splitext(basename(parsed_args["catalog_csv"]))[1]
output_filename = joinpath(OUTPUT_DIRECTORY, @sprintf("%s_synthetic.fits", catalog_label))
AccuracyBenchmark.save_images_to_fits(output_filename, generated_images)
AccuracyBenchmark.append_hash_to_file(output_filename)
