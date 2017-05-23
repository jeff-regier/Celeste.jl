#!/usr/bin/env julia

import Celeste: AccuracyBenchmark
import Celeste: ArgumentParse
import Celeste: SDSSIO

const OUTPUT_DIRECTORY = joinpath(splitdir(Base.source_path())[1], "output")

parser = ArgumentParse.ArgumentParser()
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

srand(12345)

rcf = SDSSIO.RunCamcolField(
    parsed_args["run"],
    parsed_args["camcol"],
    parsed_args["field"],
)
images = SDSSIO.load_field_images([rcf], AccuracyBenchmark.SDSS_DATA_DIR)
@assert length(images) == 5

if !isdir(OUTPUT_DIRECTORY)
    mkdir(OUTPUT_DIRECTORY)
end
filename = joinpath(
    OUTPUT_DIRECTORY,
    @sprintf("sdss_%s_%s_%s_images.fits", rcf.run, rcf.camcol, rcf.field),
)
AccuracyBenchmark.save_images_to_fits(filename, images)
AccuracyBenchmark.append_hash_to_file(filename)
