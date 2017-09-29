#!/usr/bin/env julia

using DataFrames

import Celeste.AccuracyBenchmark
import Celeste.ArgumentParse
import Celeste.Model
import Celeste.Synthetic
import Celeste.SDSSIO
import JLD

const OUTPUT_DIRECTORY = joinpath(splitdir(Base.source_path())[1], "output")

parser = Celeste.ArgumentParse.ArgumentParser()
ArgumentParse.add_argument(
    parser,
    "catalog_csv",
    help="Ground truth CSV catalog",
)
parsed_args = ArgumentParse.parse_args(parser, ARGS)

# load catalog
catalog_data = AccuracyBenchmark.read_catalog(parsed_args["catalog_csv"])
catalog_entries = [
    AccuracyBenchmark.make_catalog_entry(row)
        for row in eachrow(catalog_data)]

# load template iamges
strategy = SDSSIO.PlainFITSStrategy(AccuracyBenchmark.SDSS_DATA_DIR)
images = SDSSIO.load_field_images(strategy, [AccuracyBenchmark.STRIPE82_RCF])

# overwrite the pixels with synthetic images
Synthetic.gen_images!(images, catalog_entries)

# save images
catalog_label = splitext(basename(parsed_args["catalog_csv"]))[1]
output_filename = joinpath(OUTPUT_DIRECTORY, @sprintf("%s_synthetic.jld", catalog_label))
JLD.save(output_filename, "images", images)
AccuracyBenchmark.append_hash_to_file(output_filename)
