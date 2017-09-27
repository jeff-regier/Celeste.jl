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
    "run",
    help="SDSS run # for templates images",
    arg_type=Int,
)
ArgumentParse.add_argument(
    parser,
    "camcol",
    help="SDSS camcol # for templates images",
    arg_type=Int,
)
ArgumentParse.add_argument(
    parser,
    "field",
    help="SDSS field # for templates images",
    arg_type=Int,
)
ArgumentParse.add_argument(
    parser,
    "catalog_csv",
    help="Ground truth CSV catalog",
)
parsed_args = ArgumentParse.parse_args(parser, ARGS)

# load catalog
catalog_data = AccuracyBenchmark.read_catalog(parsed_args["catalog_csv"])
no_na_cols = :flux_r_nmgy, :color_ug, :color_gr, :color_ri, :color_iz
for col in no_na_cols
    catalog_data = catalog_data[.!isna.(catalog_data[col]), :]
end
catalog_entries = [
    AccuracyBenchmark.make_catalog_entry(row)
        for row in eachrow(catalog_data)]

# load template iamges
rcf = SDSSIO.RunCamcolField(
    parsed_args["run"],
    parsed_args["camcol"],
    parsed_args["field"],
)
strategy = SDSSIO.PlainFITSStrategy(AccuracyBenchmark.SDSS_DATA_DIR)
images = SDSSIO.load_field_images(strategy, [rcf])

# generate synthetic images
Synthetic.gen_images!(images, catalog_entries)

# save images
catalog_label = splitext(basename(parsed_args["catalog_csv"]))[1]
output_filename = joinpath(OUTPUT_DIRECTORY, @sprintf("%s_synthetic.jld", catalog_label))
JLD.save(output_filename, "images", images)
AccuracyBenchmark.append_hash_to_file(output_filename)
