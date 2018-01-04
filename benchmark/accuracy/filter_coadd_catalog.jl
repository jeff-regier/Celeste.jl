#!/usr/bin/env julia

# This script takes a catalog in csv format, removes rows containing missing values
# for flux and color, and writes out the result in csv format.
# The original catalog (with NAs) is useful for initializing Celeste on coadd
# images. The filtered catalog is useful for scoring Celeste on coadd images.
# The filtered catalog is also useful for generating synthetic images based on
# a real catalog, and then initializing and scoring Celeste on those synthetic
# images.

using DataFrames

import Celeste.AccuracyBenchmark
import Celeste.ArgumentParse

parser = Celeste.ArgumentParse.ArgumentParser()
ArgumentParse.add_argument(
    parser,
    "catalog_csv",
    help="Unfiltered CSV catalog",
)
parsed_args = ArgumentParse.parse_args(parser, ARGS)

@printf("Reading '%s'...\n", parsed_args["catalog_csv"])
catalog_data = AccuracyBenchmark.read_catalog(parsed_args["catalog_csv"])
no_na_cols = :flux_r_nmgy, :color_ug, :color_gr, :color_ri, :color_iz
for col in no_na_cols
    catalog_data = catalog_data[.!ismissing.(catalog_data[col]), :]
end

catalog_label = splitext(parsed_args["catalog_csv"])[1]
output_filename = @sprintf("%s_filtered.csv", catalog_label)
@printf("Writing '%s'...\n", output_filename)
AccuracyBenchmark.write_catalog(output_filename, catalog_data)
