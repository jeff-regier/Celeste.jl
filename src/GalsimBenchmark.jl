module GalsimBenchmark
using DataFrames
import FITSIO
import StaticArrays
import WCS

import Celeste: Model, DeterministicVI, ParallelRun, Infer

const GALSIM_BENCHMARK_DIR = joinpath(Pkg.dir("Celeste"), "benchmark", "galsim")
const FILENAME = "output/galsim_test_images.fits"

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

immutable FitsExtension
    pixels::Matrix{Float32}
    header::FITSIO.FITSHeader
end

function read_fits(filename; read_sdss_psf=false)
    println("Reading $filename...")
    @assert isfile(filename)
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
    inter_band_ratios = exp(
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
    exp(
        prior_parameters.r_mean[source_type_index]
        - prior_parameters.r_var[source_type_index]
    )
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

function actual_values(ids, star_galaxy_index, params)
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

function get_expected_dataframe(header)
    DataFrame(
        label=fill(header["CL_DESCR"], length(BENCHMARK_PARAMETER_LABELS)),
        field=BENCHMARK_PARAMETER_LABELS,
        expected=Any[
            get_field(header, "CL_X1"),
            get_field(header, "CL_Y1"),
            get_field(header, "CL_RTIO1"),
            get_field(header, "CL_ANGL1"),
            get_field(header, "CL_RADP1"),
            get_field(header, "CL_FLUX1"),
            get_field(header, "CL_C12_1"),
            get_field(header, "CL_C23_1"),
            get_field(header, "CL_C34_1"),
            get_field(header, "CL_C45_1"),
            header["CL_TYPE1"] == "star" ? 0 : 1,
        ])    
end

function benchmark_comparison_data(single_infer_params, joint_infer_params, header)
    ids = Model.ids
    star_galaxy_index = header["CL_TYPE1"] == "star" ? 1 : 2
    comparison_dataframe = get_expected_dataframe(header)
    comparison_dataframe[:single_infer_actual] = 
        actual_values(ids, star_galaxy_index, single_infer_params)
    comparison_dataframe[:joint_infer_actual] =
        actual_values(ids, star_galaxy_index, joint_infer_params)
    comparison_dataframe
end

function assert_counts_match_expected_flux(band_pixels::Vector{Matrix{Float32}},
                                           header::FITSIO.FITSHeader,
                                           iota::Float64)
    if !header["CL_NOISE"]
        expected_flux_nmgy = prod(size(band_pixels[3])) * header["CL_SKY"]
        for source_index in 1:header["CL_NSRC"]
            expected_flux_nmgy += header[string("CL_FLUX", source_index)]
        end
        expected_flux_counts = expected_flux_nmgy * iota
        @assert abs(sum(band_pixels[3]) - expected_flux_counts) / expected_flux_counts < 1e-3
    end
end

function make_catalog_entry(x_position_world_coords, y_position_world_coords)
    Model.CatalogEntry(
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

function make_catalog_entries(header::FITSIO.FITSHeader)
    catalog_entries = Model.CatalogEntry[]
    num_sources = header["CL_NSRC"]
    for source_index in 1:num_sources
        if num_sources == 1
            initial_position = [0.005335 for i in 1:2] # center of image
        else
            initial_position = [
                header[string("CL_X", source_index)],
                header[string("CL_Y", source_index)],
            ]
        end
        push!(catalog_entries, make_catalog_entry(initial_position[1], initial_position[2]))
    end
    catalog_entries
end

function parallel_inference(band_images, catalog_entries; joint_infer=false)
    # Target sources is all sources
    target_sources = collect(1:length(catalog_entries))

    # Create the neighbor map (everyone is a neighbor of each other)
    neighbor_map = Infer.find_neighbors(target_sources, catalog_entries, band_images)

    # Optimize
    results = ParallelRun.parallel_infer(catalog_entries, target_sources, neighbor_map, band_images;
                                         joint_infer=joint_infer, joint_infer_n_iters=20)
    results[1].vs
end

# Returns a data frame with one row for each test case and  parameter name, with columns
# * label: test case name
# * field: human-readable parameter name
# * expected: "ground truth" expected value
# * single_infer_actual: inferred MAP value or posterior median from single-source inference
# * joint_infer_actual: ditto, for multi-source joint inference
function main(; test_case_names=String[], print_fn=println)
    all_benchmark_data = []
    extensions, wcs = read_fits(joinpath(GALSIM_BENCHMARK_DIR, FILENAME))
    @assert length(extensions) % 5 == 0 # one extension per band for each test case

    for test_case_index in 1:div(length(extensions), 5)
        first_band_index = (test_case_index - 1) * 5 + 1
        header = extensions[first_band_index].header
        this_test_case_name = header["CL_DESCR"]
        if !isempty(test_case_names) && !in(this_test_case_name, test_case_names)
            continue
        end
        println("Running test case '$this_test_case_name'")
        iota = header["CL_IOTA"]
        psf = make_psf(header["CL_SIGMA"])

        band_pixels = [
            extensions[index].pixels for index in first_band_index:(first_band_index + 4)
        ]
        assert_counts_match_expected_flux(band_pixels, header, iota)
        band_images = make_images(band_pixels, psf, wcs, header["CL_SKY"], iota)
        catalog_entries::Vector{Model.CatalogEntry} = make_catalog_entries(header)

        single_infer_variational_parameters = parallel_inference(band_images, catalog_entries,
                                                                 joint_infer=false)
        joint_infer_variational_parameters = parallel_inference(band_images, catalog_entries,
                                                                joint_infer=true)
        
        benchmark_data = benchmark_comparison_data(single_infer_variational_parameters,
                                                   joint_infer_variational_parameters,
                                                   header)
        print_fn(repr(benchmark_data))
        push!(all_benchmark_data, benchmark_data)
    end

    full_data = vcat(all_benchmark_data...)
    print_fn(repr(full_data[!isna(full_data[:expected]), :]))
    full_data
end

end # module GalsimBenchmark
