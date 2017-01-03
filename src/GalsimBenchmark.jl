__precompile__()

module GalsimBenchmark

using DataFrames
using Distributions
import FITSIO
import StaticArrays
import WCS

import Celeste: Model, DeterministicVI, ParallelRun, Infer
import Celeste.Model: CatalogEntry
import Celeste.ParallelRun: one_node_single_infer, one_node_joint_infer

const GALSIM_BENCHMARK_DIR = joinpath(Pkg.dir("Celeste"), "benchmark", "galsim")
const LATEST_FITS_FILENAME_DIR = joinpath(GALSIM_BENCHMARK_DIR, "latest_filenames")

type GalsimFitsFileNotFound <: Exception end

function make_psf(psf_sigma_px)
    alphaBar = [1.; 0.]
    xiBar = [0.; 0.]
    tauBar = [psf_sigma_px^2 0.; 0. psf_sigma_px^2]
    [
        Model.PsfComponent(
            alphaBar[k],
            StaticArrays.SVector{2, Float64}(xiBar),
            StaticArrays.SMatrix{2, 2, Float64, 4}(tauBar)
        )
        for k in 1:2
    ]
end

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

immutable FitsExtension
    pixels::Matrix{Float32}
    header::FITSIO.FITSHeader
end

function read_fits(filename; read_sdss_psf=false)
    println("Reading '$filename'...")
    if !isfile(filename)
        println(string(
            "FITS file '$filename' not found. Try running 'make fetch' in the 'benchmark/galsim' ",
            "directory."
        ))
        throw(GalsimFitsFileNotFound())
    end
    fits = FITSIO.FITS(filename)
    println("Found $(length(fits)) extensions.")

    extensions::Vector{FitsExtension} = []
    for extension in fits
        push!(extensions, FitsExtension(read(extension), FITSIO.read_header(extension)))
    end

    # assume WCS same for each extension
    wcs = WCS.from_header(FITSIO.read_header(fits[1], String))[1]

    close(fits)
    extensions, wcs
end

function load_galsim_fits(label)
    latest_fits_filename = get_latest_fits_filename(label)
    extensions, wcs = read_fits(joinpath(GALSIM_BENCHMARK_DIR, "output", latest_fits_filename))
    @assert length(extensions) % 5 == 0 # one extension per band for each test case
    extensions, wcs
end

function make_images(band_pixels, psf, wcs, epsilon, iota)
    # assume dimensions equal for all images
    height, width = size(band_pixels[1])
    [
        Model.Image(
            height,
            width,
            band_pixels[band],
            band,
            wcs,
            psf,
            0, # SDSS run
            0, # SDSS camcol
            0, # SDSS field
            fill(epsilon, height, width),
            fill(iota, height),
            Model.RawPSF(Array(Float64, 0, 0), 0, 0, Array(Float64, 0, 0, 0)),
        )
        for band in 1:5
    ]
end

function typical_band_relative_intensities(is_star::Bool)
    source_type_index = is_star ? 1 : 2
    prior_parameters::Model.PriorParams = Model.load_prior()
    # Band relative intensities are a mixture of lognormals. Which mixture component has the most
    # weight?
    dominant_component = indmax(prior_parameters.k[:, source_type_index])
    # What are the most typical log relative intensities for that component?
    inter_band_ratios = exp.(
        prior_parameters.c_mean[:, dominant_component, source_type_index]
        - diag(prior_parameters.c_cov[:, :, dominant_component, source_type_index])
    )
    Float64[
        1 / inter_band_ratios[2] / inter_band_ratios[1],
        1 / inter_band_ratios[2],
        1,
        inter_band_ratios[3],
        inter_band_ratios[3] * inter_band_ratios[4],
    ]
end

function typical_reference_brightness(is_star::Bool)
    source_type_index = is_star ? 1 : 2
    prior_parameters::Model.PriorParams = Model.load_prior()

    # this is the mode. brightness is log normal.
    exp(prior_parameters.r_μ[source_type_index]
            - prior_parameters.r_σ²[source_type_index])
end

# Since we're considering elliptical galaxy shapes, angle is only meaningful up to rotations of 180
# deg. This finds an equivalent angle in [0, 180) deg.
function canonical_angle(params)
    angle_radians = params[Model.ids.e_angle]
    while angle_radians < 0
        angle_radians += pi
    end
    while angle_radians > pi
        angle_radians -= pi
    end
    angle_radians
end

const BENCHMARK_PARAMETER_LABELS = String[
    "X center (world coords)",
    "Y center (world coords)",
    "Minor/major axis ratio",
    "Angle (degrees)",
    "Half-light radius (pixels)",
    "Brightness (nMgy)",
    "Color band 1-2 ratio",
    "Color band 2-3 ratio",
    "Color band 3-4 ratio",
    "Color band 4-5 ratio",
    "Probability of galaxy",
]

function get_field(header::FITSIO.FITSHeader, label::String, index::Int64)
    key = @sprintf("%s%03d", label, index)
    if haskey(header, key)
        header[key]
    else
        NA
    end
end

function inferred_values(star_galaxy_index, params)
    ids = Model.ids
    Float64[
        params[ids.u[1]],
        params[ids.u[2]],
        params[ids.e_axis],
        canonical_angle(params) * 180 / pi,
        params[ids.e_scale] * sqrt(params[ids.e_axis]),
        exp(params[ids.r1[star_galaxy_index]]),
        exp(params[ids.c1[1, star_galaxy_index]]),
        exp(params[ids.c1[2, star_galaxy_index]]),
        exp(params[ids.c1[3, star_galaxy_index]]),
        exp(params[ids.c1[4, star_galaxy_index]]),
        params[ids.a[2]],
    ]
end

function get_ground_truth_dataframe(header, source_index)
    source_field(label) = get_field(header, label, source_index)
    ground_truth = @data(Any[
        source_field("CLX"),
        source_field("CLY"),
        source_field("CLRTO"),
        source_field("CLANG"),
        source_field("CLRDP"),
        source_field("CLFLX"),
        source_field("CLC12"),
        source_field("CLC23"),
        source_field("CLC34"),
        source_field("CLC45"),
        source_field("CLTYP") == "star" ? 0 : 1,
    ])
    DataFrame(
        label=fill(header["CLDESCR"], length(BENCHMARK_PARAMETER_LABELS)),
        source=fill(source_index, length(BENCHMARK_PARAMETER_LABELS)),
        field=BENCHMARK_PARAMETER_LABELS,
        ground_truth=convert(DataVector{Float64}, ground_truth),
    )
end

function error_in_posterior_std_devs(star_galaxy_index, params, header, source_index)
    ids = Model.ids
    lognormal_mean_id = vcat(
        [ids.r1[star_galaxy_index]],
        [ids.c1[band, star_galaxy_index] for band in 1:4],
    )
    lognormal_var_id = vcat(
        [ids.r2[star_galaxy_index]],
        [ids.c2[band, star_galaxy_index] for band in 1:4],
    )
    ground_truth_label = ["CLFLX", "CLC12", "CLC23", "CLC34", "CLC45"]

    posterior_z_scores = Float64[]
    for index in 1:length(lognormal_mean_id)
        ground_truth = get_field(header, ground_truth_label[index], source_index)
        error = abs(params[lognormal_mean_id[index]] - log(ground_truth))
        posterior_sd = sqrt(params[lognormal_var_id[index]])
        push!(posterior_z_scores, error / posterior_sd)
    end

    vcat(repeat(@data(Float64[NA]), outer=5), posterior_z_scores, @data(Float64[NA]))
end

function benchmark_comparison_data(inferred_params, header, source_index)
    star_galaxy_index = get_field(header, "CLTYP", source_index) == "star" ? 1 : 2
    comparison_dataframe = get_ground_truth_dataframe(header, source_index)
    comparison_dataframe[:estimate] = inferred_values(star_galaxy_index, inferred_params)
    comparison_dataframe[:error_sds] = error_in_posterior_std_devs(
        star_galaxy_index,
        inferred_params,
        header,
        source_index,
    )
    comparison_dataframe[!isna(comparison_dataframe[:ground_truth]), :]
end

function assert_counts_match_expected_flux(band_pixels::Vector{Matrix{Float32}},
                                           header::FITSIO.FITSHeader,
                                           iota::Float64)
    if !header["CLNOISE"]
        expected_flux_nmgy = prod(size(band_pixels[3])) * header["CLSKY"]
        for source_index in 1:header["CLNSRC"]
            expected_flux_nmgy += get_field(header, "CLFLX", source_index)
        end
        expected_flux_counts = expected_flux_nmgy * iota
        @assert abs(sum(band_pixels[3]) - expected_flux_counts) / expected_flux_counts < 1e-3
    end
end

function make_catalog_entry(x_position_world_coords, y_position_world_coords)
    CatalogEntry(
        [x_position_world_coords, y_position_world_coords],
        false, # is_star
        # sample_star_fluxes
        typical_band_relative_intensities(true) .* typical_reference_brightness(true),
        # sample_galaxy_fluxes
        typical_band_relative_intensities(false) .* typical_reference_brightness(false),
        0.1, # gal_frac_dev
        0.7, # gal_ab
        pi / 4, # gal_angle
        4., # gal_scale
        "sample", # objid
        0, # thing_id
    )
end

function make_catalog(header::FITSIO.FITSHeader)
    catalog = CatalogEntry[]
    num_sources = header["CLNSRC"]
    degrees_per_pixel = header["CLRES"]
    for source_index in 1:num_sources
        position_offset = rand(Uniform(-degrees_per_pixel, degrees_per_pixel), 2)
        catalog_entry = make_catalog_entry(
            get_field(header, "CLX", source_index) + position_offset[1],
            get_field(header, "CLY", source_index) + position_offset[2],
        )
        push!(catalog, catalog_entry)
    end
    catalog
end

function make_images_and_catalog(start_index, fits_extensions, wcs, header)
    psf = make_psf(header["CLSIGMA"])
    iota = header["CLIOTA"]
    band_pixels = [
        fits_extensions[index].pixels for index in start_index:(start_index + 4)
    ]
    assert_counts_match_expected_flux(band_pixels, header, iota)
    images = make_images(band_pixels, psf, wcs, header["CLSKY"], iota)
    catalog = make_catalog(header)
    images, catalog
end

# Returns a data frame with one row for each test case and parameter name, with columns
# * label: test case name
# * field: human-readable parameter name
# * ground_truth: value used in GalSim image generation
# * estimate: inferred MAP value or posterior median from single-source inference
# * error_sds: absolute error of estimate, divided by posterior standard deviation
function run_benchmarks(; test_case_names=String[], print_fn=println, joint_inference=false,
                        infer_source_callback=DeterministicVI.infer_source)
    extensions, wcs = load_galsim_fits("galsim_benchmarks")
    all_benchmark_data = []
    for test_case_index in 1:div(length(extensions), 5)
        first_band_index = (test_case_index - 1) * 5 + 1
        header = extensions[first_band_index].header
        this_test_case_name = header["CLDESCR"]
        if !isempty(test_case_names) && !in(this_test_case_name, test_case_names)
            continue
        end
        println("Running test case '$this_test_case_name'")
        num_sources = header["CLNSRC"]

        images, catalog = make_images_and_catalog(first_band_index, extensions, wcs, header)

        inferred_params = []
        if joint_inference
            target_sources = collect(1:num_sources)
            neighbor_map = Infer.find_neighbors(target_sources, catalog, images)
            results = one_node_joint_infer(catalog, target_sources, neighbor_map, images)
            inferred_params = [results[source_index].vs for source_index in 1:num_sources]
        else
            for source_index in 1:num_sources
                target_sources = [source_index,]
                neighbor_map = Infer.find_neighbors(target_sources, catalog, images)
                results = one_node_single_infer(
                    catalog,
                    target_sources,
                    neighbor_map,
                    images,
                    infer_source_callback=infer_source_callback,
                )
                @assert length(results) == 1
                push!(inferred_params, results[1].vs)
            end
        end

        for source_index in 1:num_sources
            benchmark_data = benchmark_comparison_data(
                inferred_params[source_index],
                header,
                source_index,
            )
            print_fn(repr(benchmark_data))
            push!(all_benchmark_data, benchmark_data)
        end
    end

    vcat(all_benchmark_data...)
end

function run_field()
    extensions, wcs = load_galsim_fits("galsim_field")
    header = extensions[1].header
    num_sources = header["CLNSRC"]

    images, catalog = make_images_and_catalog(1, extensions, wcs, header)
    target_sources = collect(1:num_sources)
    neighbor_map = Infer.find_neighbors(target_sources, catalog, images)
    infer_results = one_node_joint_infer(catalog, target_sources, neighbor_map, images)

    all_benchmark_data = DataFrame[]
    for source_index in 1:num_sources
        benchmark_data = benchmark_comparison_data(
            infer_results[source_index].vs,
            header,
            source_index,
        )
        push!(all_benchmark_data, benchmark_data)
    end
    vcat(all_benchmark_data...)
end

end # module GalsimBenchmark
