__precompile__()

module GalsimBenchmark

using DataFrames
using Distributions
import FITSIO
import StaticArrays
import WCS

import Celeste: AccuracyBenchmark, Infer

const GALSIM_BENCHMARK_DIR = joinpath(Pkg.dir("Celeste"), "benchmark", "galsim")
const LATEST_FITS_FILENAME_DIR = joinpath(GALSIM_BENCHMARK_DIR, "latest_filenames")
const ACTIVE_PIXELS_MIN_RADIUS_PX = Nullable(40.0)

function get_latest_fits_filename(label)
    latest_fits_filename_holder = joinpath(
        LATEST_FITS_FILENAME_DIR,
        @sprintf("latest_%s.txt", label),
    )
    println("Looking for latest FITS filename in '$latest_fits_filename_holder'")
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
                NA
            end
        end
        DataFrame(
            objid=string(source_index),
            right_ascension_deg=source_field("CLX"),
            declination_deg=source_field("CLY"),
            is_star=(source_field("CLTYP") == "star" ? 1 : 0),
            reference_band_flux_nmgy=convert(Float64, source_field("CLFLX")),
            color_log_ratio_ug=log(source_field("CLC12")),
            color_log_ratio_gr=log(source_field("CLC23")),
            color_log_ratio_ri=log(source_field("CLC34")),
            color_log_ratio_iz=log(source_field("CLC45")),
            de_vaucouleurs_mixture_weight=source_field("CLDEV"),
            minor_major_axis_ratio=source_field("CLRTO"),
            half_light_radius_px=source_field("CLRDP"),
            angle_deg=source_field("CLANG"),
        )
    end
end

function truth_comparison_df(truth_df::DataFrame, prediction_df::DataFrame)
    @assert size(truth_df, 1) == size(prediction_df, 1)
    parameter_columns = names(truth_df)
    deleteat!(parameter_columns, findin(parameter_columns, [:objid]))
    long_truth_df = stack(truth_df, parameter_columns)
    sort!(long_truth_df, cols=[:objid, :variable])
    long_prediction_df = stack(prediction_df, parameter_columns)
    sort!(long_prediction_df, cols=[:objid, :variable])
    rename!(long_truth_df, :value, :truth)
    long_truth_df[:estimate] = long_prediction_df[:value]
    long_truth_df[:error] = long_truth_df[:estimate] .- long_truth_df[:truth]
    long_truth_df
end

function run_benchmarks(; test_case_names=String[], print_fn=println, joint_inference=false,
                        use_fft=false)
    function infer_source_min_radius(args...; kwargs...)
        infer_source_callback(args...; min_radius_pix=ACTIVE_PIXELS_MIN_RADIUS_PX, kwargs...)
    end

    latest_fits_filename = get_latest_fits_filename("galsim_benchmarks")
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
        println("Running test case '$this_test_case_name'")
        num_sources = header["CLNSRC"]

        images = AccuracyBenchmark.make_images(extensions[first_band_index:(first_band_index+4)])
        truth_catalog_df = extract_catalog_from_header(header)
        catalog_entries = AccuracyBenchmark.make_initialization_catalog(truth_catalog_df, false)
        target_sources = collect(1:num_sources)
        results = AccuracyBenchmark.run_celeste(
            catalog_entries,
            target_sources,
            images,
            use_joint_inference=joint_inference,
            use_fft=use_fft,
        )
        prediction_df = AccuracyBenchmark.celeste_to_df(results)

        println(repr(truth_comparison_df(truth_catalog_df, prediction_df)))
        push!(truth_dfs, truth_catalog_df)
        push!(prediction_dfs, prediction_df)
    end
    vcat(truth_dfs...), vcat(prediction_dfs...)
end

end # module GalsimBenchmark
