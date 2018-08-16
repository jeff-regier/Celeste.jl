#!/usr/bin/env julia
# current command for Stripe 82 Experiment
# julia run_celeste_on_field_mcmc.jl \
#   --ais-output-dir ais-output-s82 \
#   --use-full-initialization \
#   --initialization-catalog ~/Proj/Celeste.jl/benchmark/accuracy/output/sdss_4263_5_119_primary_b97a8fda22.csv \
#   --target-source-range $SOURCE_RANGE
import Celeste.ArgumentParse
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
    "--target-source-range",
    help="Target only the given number of sources, for quicker testing",
    default="5:10",
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
ArgumentParse.add_argument(
    parser,
    "--ais-output-dir",
    help="location to save AIS+MCMC samples",
)
ArgumentParse.add_argument(
    parser,
    "--num_procs",
    help="number of processers to use --- a node has 64"
)
parsed_args = ArgumentParse.parse_args(parser, ARGS)
println(parsed_args)

# Set up AIS-MCMC sample output directory
ais_output_dir = "ais-output"
if haskey(parsed_args, "ais-output-dir")
    ais_output_dir = parsed_args["ais-output-dir"]
end
if !isdir(ais_output_dir)
    mkdir(ais_output_dir)
end

# handle multiple cores (alternatively --- run `julia -p <n_procs>`)
num_procs = 63
#if haskey(parsed_args, "num-procs")
#    num_procs = parse(Int, parsed_args["num_procs"])
#end
println("------ using ", num_procs, " cores -----------")
addprocs(num_procs)

# this insane syntax broadcasts the local variable parsed_args to every
# every process that we added above
@eval @everywhere parsed_args = $parsed_args
@eval @everywhere ais_output_dir = $ais_output_dir

@everywhere begin

############################
# do imports everywhere    #
############################
using DataFrames
import JLD
import Celeste
import Celeste: Config, Model, MCMC
import Celeste.ParallelRun
import Celeste.SDSSIO
import Celeste.Log
import Celeste.AccuracyBenchmark
const OUTPUT_DIRECTORY = joinpath(splitdir(Base.source_path())[1], "output")


###################################################################
# Parse input images --- if none, assume reference STRIPE82_RCF   #
###################################################################
if haskey(parsed_args, "images-jld")
    images = JLD.load(parsed_args["images-jld"], "images")
    catalog_label = splitext(basename(parsed_args["images-jld"]))[1]
else
    dataset = SDSSIO.SDSSDataSet(AccuracyBenchmark.SDSS_DATA_DIR)
    strategy = SDSSIO.PlainFITSStrategy(AccuracyBenchmark.SDSS_DATA_DIR)
    images = SDSSIO.load_field_images(dataset, AccuracyBenchmark.STRIPE82_RCF)
    catalog_label = @sprintf("sdss_%s_%s_%s", rcf.run, rcf.camcol, rcf.field)
end
@assert length(images) == 5


###########################################################################
# parse the initialization catalog
#  For stripe 82 --- use full initialization --- for 
#  For synthetic experiment, use initialization catalog, but do not use
#   full init
###########################################################################
if !haskey(parsed_args, "initialization-catalog")
    parsed_args["initialization-catalog"] = expanduser(
      "~/Proj/Celeste.jl/benchmark/accuracy/output/sdss_4263_5_119_primary_b97a8fda22.csv")
end
if haskey(parsed_args, "initialization-catalog")
    catalog_data = AccuracyBenchmark.read_catalog(parsed_args["initialization-catalog"])
    if parsed_args["use-full-initialization"]
        println("Using full initialization from ", parsed_args["initialization-catalog"])
    end
    catalog_entries = AccuracyBenchmark.make_initialization_catalog(
        catalog_data, parsed_args["use-full-initialization"])
    target_sources = collect(1:length(catalog_entries))
end


###########################################
# Generate sky patches for all sources    #
###########################################
function match_entry_to_patches(entry, detected_patches; detected_locs=nothing)
    if detected_locs == nothing
       detected_locs = hcat([p.world_center 
                            for p in detected_patches[:, 1]]...)
    end
    dists = [ sum((detected_locs[:,i]-entry.pos).^2)
              for i in 1:size(detected_locs,2) ]
    min_dist, min_idx = findmin(dists)
    return detected_patches[min_idx, :], min_idx
end

function match_catalog_sources_to_detected_patches(catalog, detected_patches)
    # center of each detected patch
    detected_locs = hcat([p.world_center 
                          for p in detected_patches[:, 1]]...)
    matched_idxs = []
    for entry in catalog
        _, m_idx = match_entry_to_patches(entry, detected_patches; detected_locs=detected_locs)
        push!(matched_idxs, m_idx)
    end
    return detected_patches[matched_idxs,:]
end

# if "use-full-initialization" flag is caught, then use the Model sky patches function
if parsed_args["use-full-initialization"]
    println("Using full initialization --- including Model Patches")
    catalog_patches = Model.get_sky_patches(images, catalog_entries)
else
    println("Not using full initialization, inferring patches from image")
    # detect patches with new function --- match to primary catalog to make comparison
    detected_catalog, detected_patches = Celeste.detect_sources(images)
    catalog_patches = match_catalog_sources_to_detected_patches(catalog_entries, detected_patches)
end

# generate neighboring source map
box = ParallelRun.BoundingBox(-1000.0, 1000.0, -1000.0, 1000.0)
entry_in_range = entry->((box.ramin < entry.pos[1] < box.ramax) &&
                         (box.decmin < entry.pos[2] < box.decmax))
target_ids = find(entry_in_range, catalog_entries)

# find objects with patches overlapping (for this specific patch)
neighbor_map = Dict(id=>Model.find_neighbors(catalog_patches, id)
                    for id in target_ids)


#######################################
# define process function for pmap    #
#######################################
function proc(ts)
    println("========== source ", ts, " of ", length(target_sources))
    # config = px_radius, num_temperatures, num_ais_samples, num_vi_iterations
    config = Config(25.0, 200, 25, 0)

    neighbor_ids = neighbor_map[ts]
    res = ParallelRun.process_source_mcmc(
        config, ts, catalog_entries, catalog_patches, neighbor_ids, images)

    # save source samples
    println(" ... saving source ", ts)
    fname = joinpath(ais_output_dir, "source-$(lpad(ts, 4, 0)).jld")
    JLD.save(fname, "res", res)
    return res
end

end ## everywhere end


#####################################################################
# Start Script                                                      #
#####################################################################
@printf("Loaded %d sources...\n", length(catalog_entries))

if haskey(parsed_args, "target-source-range")
    src_start, src_end = split(parsed_args["target-source-range"], ":")
    target_sources = parse(src_start):parse(src_end)
    println("====== inferring sources ", target_sources, " =========")
    target_sources = collect(target_sources)
end
