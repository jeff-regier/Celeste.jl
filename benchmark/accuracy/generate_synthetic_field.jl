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

# load template images
dataset = SDSSIO.SDSSDataSet(AccuracyBenchmark.SDSS_DATA_DIR)
images = SDSSIO.load_field_images(dataset, AccuracyBenchmark.STRIPE82_RCF)

# overwrite the pixels with synthetic images
Synthetic.gen_images!(images, catalog_entries)

# save images
catalog_label = splitext(basename(parsed_args["catalog_csv"]))[1]

hash_string = hex(hash(images))[1:10]
output_filename = @sprintf("%s_synthetic_%s.jld", catalog_label, hash_string)
output_path = joinpath(OUTPUT_DIRECTORY, output_filename)
JLD.save(output_path, "images", images)
