module GalsimBenchmark

using DataFrames
import FITSIO

import ..AccuracyBenchmark
import ..Config
import ..ParallelRun
import ..Model

const GALSIM_BENCHMARK_DIR = joinpath(Pkg.dir("Celeste"), "benchmark", "galsim")
const LATEST_FITS_FILENAME_DIR = joinpath(GALSIM_BENCHMARK_DIR, "latest_filenames")
const ACTIVE_PIXELS_MIN_RADIUS_PX = 40.0

function get_latest_fits_filename(label; verbose=false)
    latest_fits_filename_holder = joinpath(
        LATEST_FITS_FILENAME_DIR,
        @sprintf("latest_%s.txt", label),
    )
    verbose && println("Looking for latest FITS filename in '$latest_fits_filename_holder'")
    open(latest_fits_filename_holder) do stream
        return strip(readstring(stream))
    end
end

function extract_catalog_from_header(header::FITSIO.FITSHeader)
    mapreduce(vcat, 1:header["CLNSRC"]) do source_index
        function source_field(label)
            key = @sprintf("%s%03d", label, source_index)
            if haskey(header, key)
                header[key]
            else
                missing
            end
        end
        DataFrame(
            objid=@sprintf("%s_%03d", header["CLDESCR"], source_index),
            ra=source_field("CLRA"),
            dec=source_field("CLDEC"),
            is_star=(source_field("CLTYP") == "star" ? 1 : 0),
            flux_r_nmgy=convert(Float64, source_field("CLFLX")),
            color_ug=log(source_field("CLC12")),
            color_gr=log(source_field("CLC23")),
            color_ri=log(source_field("CLC34")),
            color_iz=log(source_field("CLC45")),
            gal_frac_dev=source_field("CLDEV"),
            gal_axis_ratio=source_field("CLRTO"),
            gal_radius_px=source_field("CLRDP"),
            gal_angle_deg=source_field("CLANG"),
        )
    end
end

function truth_comparison_df(truth_df::DataFrame, prediction_df::DataFrame)
    @assert size(truth_df, 1) == size(prediction_df, 1)
    parameter_columns = names(truth_df)

    # remove objid if present
    deleteat!(parameter_columns, findin(parameter_columns, [:objid]))

    # add an object index to truth and prediction, for when we reshape
    idx = collect(1:nrow(truth_df))
    truth_df = hcat(DataFrame(index=idx), truth_df)
    prediction_df = hcat(DataFrame(index=idx), prediction_df)

    long_truth_df = stack(truth_df, parameter_columns)
    sort!(long_truth_df, [:index, :variable])

    long_prediction_df = stack(prediction_df, parameter_columns)
    sort!(long_prediction_df, [:index, :variable])

    rename!(long_truth_df, :value => :truth)
    long_truth_df[:estimate] = long_prediction_df[:value]
    long_truth_df[:error] = long_truth_df[:estimate] .- long_truth_df[:truth]
    long_truth_df
end

function run_benchmarks(; test_case_names=String[], joint_inference=false,
                        verbose=false)
    latest_fits_filename = get_latest_fits_filename("galsim_benchmarks"; verbose=verbose)
    full_fits_path = joinpath(GALSIM_BENCHMARK_DIR, "output", latest_fits_filename)
    extensions = AccuracyBenchmark.read_fits(full_fits_path)

    truth_dfs, prediction_dfs = DataFrame[], DataFrame[]
    all_benchmark_data = mapreduce(vcat, 1:div(length(extensions), 5)) do test_case_index
        first_band_index = (test_case_index - 1) * 5 + 1
        header = extensions[first_band_index].header
        this_test_case_name = header["CLDESCR"]
        if !isempty(test_case_names) && !in(this_test_case_name, test_case_names)
            return DataFrame()
        end
        verbose && println("Running test case '$this_test_case_name'")
        num_sources = header["CLNSRC"]

        images = AccuracyBenchmark.make_images(extensions[first_band_index:(first_band_index+4)])
        truth_catalog_df = extract_catalog_from_header(header)
        catalog_entries = AccuracyBenchmark.make_initialization_catalog(truth_catalog_df, false)
        target_sources = collect(1:num_sources)
        config = Config(min_radius_pix = ACTIVE_PIXELS_MIN_RADIUS_PX)
        patches = Model.get_sky_patches(images, catalog_entries)
        neighbor_map = Dict(i=>Model.find_neighbors(patches, i)
                            for i in target_sources)

        if joint_inference
            results = ParallelRun.one_node_joint_infer(catalog_entries,
                                                       patches,
                                                       target_sources,
                                                       neighbor_map, images,
                                                       config=config)
        else
            results = ParallelRun.one_node_single_infer(catalog_entries,
                                                        patches,
                                                        target_sources,
                                                        neighbor_map, images,
                                                        config=config)
        end

        prediction_df = AccuracyBenchmark.celeste_to_df(results)

        verbose && println(repr(truth_comparison_df(truth_catalog_df, prediction_df)))
        push!(truth_dfs, truth_catalog_df)
        push!(prediction_dfs, prediction_df)
    end
    vcat(truth_dfs...), vcat(prediction_dfs...)
end

end # module GalsimBenchmark
