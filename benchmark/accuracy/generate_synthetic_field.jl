#!/usr/bin/env julia

using DataFrames

import Celeste.AccuracyBenchmark
import Celeste.ArgumentParse
import Celeste.Model
import Celeste.Synthetic

const PSF_SIGMA_PX = 2.29 # similar to SDSS
const OUTPUT_DIRECTORY = joinpath(splitdir(Base.source_path())[1], "output")

parser = Celeste.ArgumentParse.ArgumentParser()
ArgumentParse.add_argument(
    parser,
    "--sky_level_nmgy",
    arg_type=Float64,
    default=0.155,
    help="Sky background noise level, nMgy",
)
ArgumentParse.add_argument(
    parser,
    "--nelec_per_nmgy",
    arg_type=Float64,
    default=180.0,
    help="Nelec (units of pixel values) per nMgy of flux (aka iota)",
)
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
    parsed_args["sky_level_nmgy"],
    parsed_args["nelec_per_nmgy"],
)
generated_images = Synthetic.gen_blob(template_images, catalog_entries)

catalog_label = splitext(basename(parsed_args["catalog_csv"]))[1]
output_filename = joinpath(OUTPUT_DIRECTORY, @sprintf("%s_synthetic.fits", catalog_label))
AccuracyBenchmark.save_images_to_fits(output_filename, generated_images)
AccuracyBenchmark.append_hash_to_file(output_filename)
