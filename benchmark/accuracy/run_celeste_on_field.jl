#!/usr/bin/env julia

using DataFrames

import Celeste.AccuracyBenchmark
import Celeste.ArgumentParse
import Celeste: Config
import Celeste.ParallelRun
import Celeste.SDSSIO

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
    "--min-radius-px",
    help="Override Config.min_radius_px with the given value",
    arg_type=Float64,
)
ArgumentParse.add_argument(
    parser,
    "--image-fits",
    help="FITS file containing synthetic imagery; if not specified, will use SDSS RCF",
)
ArgumentParse.add_argument(
    parser,
    "--run",
    help="SDSS run #",
    arg_type=Int,
    default=AccuracyBenchmark.STRIPE82_RCF.run,
)
ArgumentParse.add_argument(
    parser,
    "--camcol",
    help="SDSS camcol #",
    arg_type=Int,
    default=AccuracyBenchmark.STRIPE82_RCF.camcol,
)
ArgumentParse.add_argument(
    parser,
    "--field",
    help="SDSS field #",
    arg_type=Int,
    default=AccuracyBenchmark.STRIPE82_RCF.field,
)
ArgumentParse.add_argument(
    parser,
    "catalog_csv",
    help="CSV catalog for initialization",
    required=true,
)
parsed_args = ArgumentParse.parse_args(parser, ARGS)

srand(12345)

if haskey(parsed_args, "image-fits")
    extensions = AccuracyBenchmark.read_fits(parsed_args["image-fits"])
    images = AccuracyBenchmark.make_images(extensions)
    catalog_label = splitext(basename(parsed_args["image-fits"]))[1]
else
    rcf = SDSSIO.RunCamcolField(
        parsed_args["run"],
        parsed_args["camcol"],
        parsed_args["field"],
    )
    strategy = SDSSIO.PlainFITSStrategy(AccuracyBenchmark.SDSS_DATA_DIR)
    images = SDSSIO.load_field_images(strategy, [rcf])
    catalog_label = @sprintf("sdss_%s_%s_%s", rcf.run, rcf.camcol, rcf.field)
end
@assert length(images) == 5

catalog_data = AccuracyBenchmark.read_catalog(parsed_args["catalog_csv"])
if parsed_args["use-full-initialization"]
    @printf("Using full initialization from %s\n", parsed_args["catalog_csv"])
end
catalog_entries = AccuracyBenchmark.make_initialization_catalog(
    catalog_data,
    parsed_args["use-full-initialization"],
)
@printf("Loaded %d sources...\n", length(catalog_entries))

if haskey(parsed_args, "limit-num-sources")
    target_sources = collect(1:parsed_args["limit-num-sources"])
else
    target_sources = collect(1:length(catalog_entries))
end

config = Config()
if haskey(parsed_args, "min-radius-px")
    config.min_radius_pix = parsed_args["min-radius-px"]
end
results = AccuracyBenchmark.run_celeste(
    config,
    catalog_entries,
    target_sources,
    images,
    use_joint_inference=parsed_args["joint"],
)
results_df = AccuracyBenchmark.celeste_to_df(results)

# save results in both JLD and CSV format

jld_filename = joinpath(
    OUTPUT_DIRECTORY,
    @sprintf(
        "celeste-%.4f-%.4f-%.4f-%.4f.jld",
        minimum(results_df[:ra]),
        maximum(results_df[:ra]),
        minimum(results_df[:dec]),
        maximum(results_df[:dec]),
    )
)
println("Writing $(jld_filename)...")
JLD.save(jld_filename, "results", results)

csv_filename = joinpath(OUTPUT_DIRECTORY, @sprintf("%s_predictions.csv", catalog_label))
AccuracyBenchmark.write_catalog(csv_filename, results_df)
AccuracyBenchmark.append_hash_to_file(csv_filename)
