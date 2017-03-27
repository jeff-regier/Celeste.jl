#!/usr/bin/env julia

using DataFrames

import Celeste.AccuracyBenchmark
import Celeste.ArgumentParse
import Celeste.Model
import Celeste.Synthetic

const PSF_SIGMA_PX = 2.29 # similar to SDSS
const COUNTS_PER_NMGY = 180.0 # a.k.a. "iota" in Celeste
const SKY_LEVEL_NMGY = 0.155
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
    SKY_LEVEL_NMGY,
    COUNTS_PER_NMGY,
)
generated_images = Synthetic.gen_blob(template_images, catalog_entries)

catalog_label = splitext(basename(parsed_args["catalog_csv"]))[1]
output_filename = joinpath(OUTPUT_DIRECTORY, @sprintf("%s_synthetic.fits", catalog_label))
AccuracyBenchmark.save_images_to_fits(output_filename, generated_images)
AccuracyBenchmark.append_hash_to_file(output_filename)
