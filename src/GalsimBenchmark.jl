module GalsimBenchmark
using DataFrames
import FITSIO
import StaticArrays
import WCS

import Celeste: Model, DeterministicVI, ParallelRun, Infer
import Celeste.Model: CatalogEntry
import Celeste.ParallelRun: one_node_single_infer, one_node_joint_infer

const GALSIM_BENCHMARK_DIR = joinpath(Pkg.dir("Celeste"), "benchmark", "galsim")
const LATEST_FITS_FILENAME_HOLDER = joinpath(
    GALSIM_BENCHMARK_DIR, "latest_filenames", "latest_galsim_benchmarks.txt"
)

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

function get_latest_fits_filename()
    println("Looking for latest FITS filename in '$LATEST_FITS_FILENAME_HOLDER'")
    open(LATEST_FITS_FILENAME_HOLDER) do stream
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

function get_field(header::FITSIO.FITSHeader, key::String)
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

function get_ground_truth_dataframe(header)
    DataFrame(
        label=fill(header["CLDESCR"], length(BENCHMARK_PARAMETER_LABELS)),
        field=BENCHMARK_PARAMETER_LABELS,
        ground_truth=Any[
            get_field(header, "CLX001"),
            get_field(header, "CLY001"),
            get_field(header, "CLRTO001"),
            get_field(header, "CLANG001"),
            get_field(header, "CLRDP001"),
            get_field(header, "CLFLX001"),
            get_field(header, "CLC12001"),
            get_field(header, "CLC23001"),
            get_field(header, "CLC34001"),
            get_field(header, "CLC45001"),
            header["CLTYP001"] == "star" ? 0 : 1,
        ]
    )
end

function error_in_posterior_std_devs(star_galaxy_index, params, header)
    ids = Model.ids
    lognormal_mean_id = vcat(
        [ids.r1[star_galaxy_index]],
        [ids.c1[band, star_galaxy_index] for band in 1:4],
    )
    lognormal_var_id = vcat(
        [ids.r2[star_galaxy_index]],
        [ids.c2[band, star_galaxy_index] for band in 1:4],
    )
    ground_truth_key = ["CLFLX001", "CLC12001", "CLC23001", "CLC34001", "CLC45001"]

    posterior_z_scores = Any[]
    for index in 1:length(lognormal_mean_id)
        error = abs(params[lognormal_mean_id[index]] - log(get_field(header, ground_truth_key[index])))
        posterior_sd = sqrt(params[lognormal_var_id[index]])
        push!(posterior_z_scores, error / posterior_sd)
    end

    vcat(fill(NA, 5), posterior_z_scores, [NA])
end

function benchmark_comparison_data(single_infer_params, joint_infer_params, header)
    star_galaxy_index = header["CLTYP001"] == "star" ? 1 : 2
    comparison_dataframe = get_ground_truth_dataframe(header)
    comparison_dataframe[:single_inferred] = inferred_values(star_galaxy_index, single_infer_params)
    comparison_dataframe[:joint_inferred] = inferred_values(star_galaxy_index, joint_infer_params)
    comparison_dataframe[:error_sds] = error_in_posterior_std_devs(
        star_galaxy_index,
        joint_infer_params,
        header,
    )
    comparison_dataframe
end

function assert_counts_match_expected_flux(band_pixels::Vector{Matrix{Float32}},
                                           header::FITSIO.FITSHeader,
                                           iota::Float64)
    if !header["CLNOISE"]
        expected_flux_nmgy = prod(size(band_pixels[3])) * header["CLSKY"]
        for source_index in 1:header["CLNSRC"]
            expected_flux_nmgy += header[@sprintf("CLFLX%03d", source_index)]
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
    for source_index in 1:num_sources
        if num_sources == 1
            initial_position = [0.005335 for i in 1:2] # center of image
        else
            initial_position = [
                header[@sprintf("CLX%03d", source_index)],
                header[@sprintf("CLY%03d", source_index)],
            ]
        end
        push!(catalog, make_catalog_entry(initial_position[1], initial_position[2]))
    end
    catalog
end

# Returns a data frame with one row for each test case and  parameter name, with columns
# * label: test case name
# * field: human-readable parameter name
# * expected: "ground truth" expected value
# * single_infer_actual: inferred MAP value or posterior median from single-source inference
# * joint_infer_actual: ditto, for multi-source joint inference
function main(; test_case_names=String[], print_fn=println,
              infer_source_callback=DeterministicVI.infer_source)
    latest_fits_filename = get_latest_fits_filename()
    extensions, wcs = read_fits(joinpath(GALSIM_BENCHMARK_DIR, "output", latest_fits_filename))
    @assert length(extensions) % 5 == 0 # one extension per band for each test case

    all_benchmark_data = []
    for test_case_index in 1:div(length(extensions), 5)
        first_band_index = (test_case_index - 1) * 5 + 1
        header = extensions[first_band_index].header
        this_test_case_name = header["CLDESCR"]
        if !isempty(test_case_names) && !in(this_test_case_name, test_case_names)
            continue
        end
        println("Running test case '$this_test_case_name'")
        iota = header["CLIOTA"]
        psf = make_psf(header["CLSIGMA"])
        n_sources = header["CLNSRC"]

        band_pixels = [
            extensions[index].pixels for index in first_band_index:(first_band_index + 4)
        ]
        assert_counts_match_expected_flux(band_pixels, header, iota)

        images = make_images(band_pixels, psf, wcs, header["CLSKY"], iota)
        catalog = make_catalog(header)

        # we're only scoring one object per image
        target_sources = [1,]
        neighbor_map = Infer.find_neighbors(target_sources, catalog, images)

        target_sources_joint = collect(1:n_sources)
        neighbor_map_joint = Infer.find_neighbors(target_sources_joint, catalog, images)

        single_results = one_node_single_infer(
            catalog, target_sources, neighbor_map, images;
            infer_source_callback=infer_source_callback)
        joint_results = one_node_joint_infer(catalog, target_sources_joint,
                                             neighbor_map_joint, images)

        benchmark_data = benchmark_comparison_data(single_results[1].vs,
                                                   joint_results[1].vs,
                                                   header)
        print_fn(repr(benchmark_data))
        push!(all_benchmark_data, benchmark_data)
    end

    full_data = vcat(all_benchmark_data...)
    print_fn(repr(full_data[!isna(full_data[:ground_truth]), :]))
    full_data
end

end # module GalsimBenchmark
