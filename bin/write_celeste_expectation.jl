#!/usr/bin/env julia

using DataFrames
import JLD

import Celeste.AccuracyBenchmark
import Celeste.ArgumentParse
import Celeste.DeterministicVI
import Celeste.Log
import Celeste.Model
import Celeste.ParallelRun
import Celeste.SDSSIO
import Celeste.Transform

const TEST_DATA_DIR = joinpath(Pkg.dir("Celeste"), "test", "data")
const FIELD_EXTENTS_FITS = joinpath(Pkg.dir("Celeste", "test", "data", "field_extents.fits"))
const CELESTE_CATALOG_DIR = "/global/cscratch1/sd/jregier/celeste_sdds_catalog_jld"
const CATALOG_FILE_REGEXP = r"^celeste-(-?[0-9\.]+)-(-?[0-9\.]+)-(-?[0-9\.]+)-(-?[0-9\.]+)\.jld$"

parser = ArgumentParse.ArgumentParser()
ArgumentParse.add_argument(
    parser,
    "--catalog-dir",
    help="Path to Celeste catalog files (celeste-<ramin>-<ramax>-<decmin>-<decmax>.jld)",
    default=CELESTE_CATALOG_DIR,
)
ArgumentParse.add_argument(
    parser,
    "--subtract-sdss",
    action=:store_true,
    help="Write Celeste expectation - SDSS imagery instead of just expectation",
)
ArgumentParse.add_argument(
    parser,
    "run",
    help="SDSS run #",
    arg_type=Int,
)
ArgumentParse.add_argument(
    parser,
    "camcol",
    help="SDSS camcol #",
    arg_type=Int,
)
ArgumentParse.add_argument(
    parser,
    "field",
    help="SDSS field #",
    arg_type=Int,
)
parsed_args = ArgumentParse.parse_args(parser, ARGS)

function find_relevant_catalog_files(
    ra_min_deg::Float64, ra_max_deg::Float64, dec_min_deg::Float64, dec_max_deg::Float64
)
    @assert ra_max_deg >= ra_min_deg
    @assert dec_max_deg >= dec_min_deg

    relevant_paths = String[]
    for filename in readdir(parsed_args["catalog-dir"])
        match_values = match(CATALOG_FILE_REGEXP, filename)
        if match_values === nothing
            continue
        else
            ra_start_deg = parse(Float64, match_values.captures[1])
            ra_end_deg = parse(Float64, match_values.captures[2])
            dec_start_deg = parse(Float64, match_values.captures[3])
            dec_end_deg = parse(Float64, match_values.captures[4])
            if ra_start_deg > ra_max_deg || ra_end_deg < ra_min_deg
                continue
            elseif dec_start_deg > dec_max_deg || dec_end_deg < dec_min_deg
                continue
            else
                push!(relevant_paths, joinpath(parsed_args["catalog-dir"], filename))
            end
        end
    end
    relevant_paths
end

function load_sources(
    ra_min_deg::Float64, ra_max_deg::Float64, dec_min_deg::Float64, dec_max_deg::Float64
)
    file_paths = find_relevant_catalog_files(ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg)
    Log.info("Found $(length(file_paths)) relevant catalog files")
    all_sources = ParallelRun.OptimizedSource[]
    for file_path in file_paths
        Log.info("Loading $file_path")
        sources::Vector{ParallelRun.OptimizedSource} = JLD.load(file_path)["results"]
        Log.info("  Found $(length(sources)) sources in $file_path")
        for source in sources
            source_ra_deg = source.vs[Model.ids.pos[1]]
            source_dec_deg = source.vs[Model.ids.pos[2]]
            if source_ra_deg < ra_min_deg || source_ra_deg > ra_max_deg
                continue
            elseif source_dec_deg < dec_min_deg || source_dec_deg > dec_max_deg
                continue
            else
                push!(all_sources, source)
            end
        end
    end
    all_sources
end

function catalog_entry_from_variational_params(variational_parameters)
    data = AccuracyBenchmark.variational_parameters_to_data_frame_row("0", variational_parameters)
    AccuracyBenchmark.make_catalog_entry(first(eachrow(data)))
end

# based on code in DeterministicVI.elbo_likelihood()
function fill_celeste_expectation!(
    images::Vector{Model.Image}, sources::Vector{ParallelRun.OptimizedSource}
)
    catalog = [catalog_entry_from_variational_params(source.vs) for source in sources]
    patches = Model.get_sky_patches(images, catalog, radius_override_pix=25.0)
    variational_params = Vector{Float64}[source.vs for source in sources]

    active_sources = [1]
    elbo_args = DeterministicVI.ElboArgs(images, patches, active_sources)
    elbo_vars = DeterministicVI.ElboIntermediateVariables(Float64, elbo_args.Sa, false, false)
    bvn_bundle = Model.BvnBundle{Float64}(elbo_args.psf_K, elbo_args.S)
    source_brightnesses = DeterministicVI.load_source_brightnesses(
        elbo_args,
        variational_params,
        calculate_gradient=false,
        calculate_hessian=false,
    )
    for image_index in 1:elbo_args.N
        Model.load_bvn_mixtures!(
            bvn_bundle.star_mcs,
            bvn_bundle.gal_mcs,
            elbo_args.S,
            elbo_args.patches,
            variational_params,
            elbo_args.active_sources,
            elbo_args.psf_K,
            image_index,
            elbo_vars.elbo.has_gradient,
            elbo_vars.elbo.has_hessian,
        )
        image = images[image_index]
        for h in 1:image.H, w in 1:image.W
            # fills elbo_vars.E_G in-place
            DeterministicVI.add_pixel_term!(
                elbo_args,
                variational_params,
                image_index, h, w,
                bvn_bundle,
                source_brightnesses,
                elbo_vars,
            )
            image.pixels[h, w] += elbo_vars.E_G.v[] - image.sky[h, w]
        end
    end
end

function subtract_sdss_values!(
    celeste_images::Vector{Model.Image}, sdss_images::Vector{Model.Image}
)
    for band in 1:5, h in 1:celeste_images[band].H, w in 1:celeste_images[band].W
        celeste_images[band].pixels[h, w] -= sdss_images[band].pixels[h, w]
    end
end

function main()
    rcf = SDSSIO.RunCamcolField(
        parsed_args["run"],
        parsed_args["camcol"],
        parsed_args["field"],
    )

    working_directory = pwd()
    cd(TEST_DATA_DIR)
    command = `make RUN=$(rcf.run) CAMCOL=$(rcf.camcol) FIELD=$(rcf.field)`
    Log.info("Downloading SDSS frame files ($command in $TEST_DATA_DIR)...")
    run(command)
    cd(working_directory)

    Log.info("Opening field extents: $FIELD_EXTENTS_FITS")
    field_extents_data = AccuracyBenchmark.load_fits_table_as_data_frame(
        FIELD_EXTENTS_FITS,
        2,
        [:run, :camcol, :field, :ramin, :ramax, :decmin, :decmax],
    )
    row_selector = (
        (field_extents_data[:run] .== rcf.run)
        .& (field_extents_data[:camcol] .== rcf.camcol)
        .& (field_extents_data[:field] .== rcf.field)
    )
    field_row = field_extents_data[row_selector, :]
    Log.info(field_row)

    full_sources = load_sources(
        field_row[1, :ramin],
        field_row[1, :ramax],
        field_row[1, :decmin],
        field_row[1, :decmax],
    )
    Log.info("Found $(length(full_sources)) sources total")

    strategy = SDSSIO.PlainFITSStrategy(AccuracyBenchmark.SDSS_DATA_DIR)
    sdss_images = SDSSIO.load_field_images(strategy, [rcf])
    images = deepcopy(sdss_images)

    for band in 1:5, h in 1:images[band].H, w in 1:images[band].W
        images[band].pixels[h, w] = 0
    end

    fill_celeste_expectation!(images, full_sources)

    label = "expectations"
    if parsed_args["subtract-sdss"]
        subtract_sdss_values!(images, sdss_images)
        label = "errors"
    end

    output_filename = "celeste_$(label)_$(rcf.run)_$(rcf.camcol)_$(rcf.field).fits"
    AccuracyBenchmark.save_images_to_fits(output_filename, images)
end

main()
