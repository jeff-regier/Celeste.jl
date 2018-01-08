# Implementation of main console executable and related I/O.

using YAML
using JLD

import .ArgumentParse
import .Model: Image
import .ParallelRun: infer_box
using .SDSSIO

# read universal and dataset-specific configurations
function read_config(filename)
    data = open(YAML.load, filename)

    # universal settings
    cfg_keys = ["min_radius_pix",
                "num_ais_temperatures",
                "num_ais_samples",
                "num_joint_vi_iters"]
    # dictionary of cfg_keys actually present in configuration file.
    cfg_dict = Dict(k=>data[k] for k in cfg_keys if haskey(data, k))
    cfg = Config(;cfg_dict...)

    # dataset-specific settings
    datasets = Dict()
    for (name, dataset_config) in data["datasets"]
        if name == "sdss"
            basedir = dataset_config["basedir"]
            iostrategy = Symbol(get(dataset_config, "iostrategy", "plain"))
            dirlayout = Symbol(get(dataset_config, "dirlayout", "celeste"))
            slurp = get(dataset_config, "slurp", false)
            compressed = get(dataset_config, "compressed", false)
            datasets[name] = SDSSDataSet(basedir;
                                         dirlayout = dirlayout,
                                         compressed = compressed,
                                         iostrategy = iostrategy,
                                         slurp = slurp)
        else
            error("unrecognized dataset name: $(name)")
        end
    end

    return cfg, datasets
end


"""
Save provided results to a JLD file.
"""
function save_results(outdir, ramin, ramax, decmin, decmax, results)
    fname = @sprintf("%s/celeste-%.4f-%.4f-%.4f-%.4f.jld",
                     outdir, ramin, ramax, decmin, decmax)
    JLD.save(fname, "results", results)
end
save_results(outdir, box::BoundingBox, results) =
    save_results(outdir, box.ramin, box.ramax, box.decmin, box.decmax, results)


# command-line script
function main(console_args::Vector{String}=ARGS)

    parser = ArgumentParse.ArgumentParser(program_name="infer-box")
    ArgumentParse.add_argument(parser, "datasets",
                               help="comma-separated dataset names. Each dataset must appear in config file.")
    ArgumentParse.add_argument(parser, "ramin", arg_type=Float64)
    ArgumentParse.add_argument(parser, "ramax", arg_type=Float64)
    ArgumentParse.add_argument(parser, "decmin", arg_type=Float64)
    ArgumentParse.add_argument(parser, "decmax", arg_type=Float64)
    ArgumentParse.add_argument(parser, "--config",
                               help="YAML configuration file.",
                               default="celeste.yml")
    ArgumentParse.add_argument(parser, "--method",
                               help="inference method: {joint_vi, single_vi, mcmc}",
                               default="joint_vi")

    args = ArgumentParse.parse_args(parser, console_args)

    # read config
    config, known_datasets = read_config(args["config"])

    # initialize data sets
    names = split(args["datasets"], ",")
    datasets = [known_datasets[name] for name in names]

    box = BoundingBox(args["ramin"], args["ramax"],
                      args["decmin"], args["decmax"])

    # load images from all datasets
    images = Image[]
    for dataset in datasets
        append!(images, load_images(dataset, box))
    end

    results = infer_box(images, box; config = config,
                        method = Symbol(args["method"]))
    save_results(".", box, results)
    return 0
end
