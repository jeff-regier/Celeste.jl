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

if haskey(parsed_args, "initialization-catalog")
    catalog_data = AccuracyBenchmark.read_catalog(parsed_args["initialization-catalog"])
    if parsed_args["use-full-initialization"]
        println("Using full initialization from ", parsed_args["initialization-catalog"])
    end
    catalog_entries = AccuracyBenchmark.make_initialization_catalog(
        catalog_data, parsed_args["use-full-initialization"])
    target_sources = collect(1:length(catalog_entries))
    neighbor_map = ParallelRun.find_neighbors(target_sources, catalog_entries,
                                              images)
else
    catalog_entries, target_sources, neighbor_map =
        ParallelRun.infer_init(images)
end

@printf("Loaded %d sources...\n", length(catalog_entries))

if haskey(parsed_args, "limit-num-sources")
    nsources = min(parsed_args["limit-num-sources"], length(target_sources))
    target_sources = target_sources[1:nsources]
end

config = Config(25.0)
if parsed_args["joint"]
    results = ParallelRun.one_node_joint_infer(catalog_entries, target_sources,
                                               neighbor_map, images,
                                               config=config)
else
    results = ParallelRun.one_node_single_infer(catalog_entries, target_sources,
                                                neighbor_map, images,
                                                config=config)
end

results_df = AccuracyBenchmark.celeste_to_df(results)

csv_filename = joinpath(OUTPUT_DIRECTORY, @sprintf("%s_predictions.csv", catalog_label))
AccuracyBenchmark.write_catalog(csv_filename, results_df)
AccuracyBenchmark.append_hash_to_file(csv_filename)
