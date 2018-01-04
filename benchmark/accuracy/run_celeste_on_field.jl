#!/usr/bin/env julia

using DataFrames
import JLD

import Celeste.AccuracyBenchmark
import Celeste.ArgumentParse
import Celeste: Config
import Celeste.ParallelRun
import Celeste.SDSSIO
import Celeste.Log

const OUTPUT_DIRECTORY = joinpath(splitdir(Base.source_path())[1], "output")

parser = Celeste.ArgumentParse.ArgumentParser()
ArgumentParse.add_argument(parser, "--joint", help="Use joint inference", action=:store_true)
ArgumentParse.add_argument(
    parser,
    "--use-full-initialization",
    help="Use all information from initialization catalog (otherwise will just use noisy position",
    action=:store_true,
)
ArgumentParse.add_argument(
    parser,
    "--limit-num-sources",
    help="Target only the given number of sources, for quicker testing",
    arg_type=Int,
)
ArgumentParse.add_argument(
    parser,
    "--images-jld",
    help="FITS file containing synthetic imagery; if not specified, will use SDSS (primary) images",
)
ArgumentParse.add_argument(
    parser,
    "--initialization-catalog",
    help="CSV catalog for initialization. Default is to initialize by source detection.",
)
parsed_args = ArgumentParse.parse_args(parser, ARGS)

if haskey(parsed_args, "images-jld")
    images = JLD.load(parsed_args["images-jld"], "images")
    catalog_label = splitext(basename(parsed_args["images-jld"]))[1]
else
    rcf = AccuracyBenchmark.STRIPE82_RCF
    strategy = SDSSIO.PlainFITSStrategy(AccuracyBenchmark.SDSS_DATA_DIR)
    images = SDSSIO.load_field_images(strategy, [rcf])
    catalog_label = @sprintf("sdss_%s_%s_%s", rcf.run, rcf.camcol, rcf.field)
end
@assert length(images) == 5


config = Config(25.0)
method = parsed_args["joint"] ? :joint : :single
box = ParallelRun.BoundingBox(-1000.0, 1000.0, -1000.0, 1000.0)

if haskey(parsed_args, "initialization-catalog")
    rawcatalog = AccuracyBenchmark.read_catalog(parsed_args["initialization-catalog"])
    if parsed_args["use-full-initialization"]
        println("Using full initialization from ", parsed_args["initialization-catalog"])
    end
    catalog = AccuracyBenchmark.make_initialization_catalog(
        rawcatalog, parsed_args["use-full-initialization"])

    # TODO: add option in main entry point (infer_box) for limiting
    # number of target sources. For two reasons: (1) enable limiting
    # sources when a catalog is not passed (below). (2) When a catalog is
    # passed, still consider neighbors even when they're not targets.
    if haskey(parsed_args, "limit-num-sources")
        nsources = min(parsed_args["limit-num-sources"],
                       length(catalog))
        catalog = catalog[1:nsources]
    end

    results = ParallelRun.infer_box(images, catalog, box;
                                    method=method, config=config)
else
    results = ParallelRun.infer_box(images, box; method=method, config=config)
end

results_df = AccuracyBenchmark.celeste_to_df(results)

csv_filename = joinpath(OUTPUT_DIRECTORY, "$(catalog_label)_predictions.csv")
csv_filename = AccuracyBenchmark.write_catalog(csv_filename, results_df; append_hash=true)
@printf("Wrote '%s'...\n", csv_filename)
