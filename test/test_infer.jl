## test the main entry point in Celeste: the `infer` function
import JLD

import Celeste: Config
using Celeste.SDSSIO
using Celeste.ParallelRun

@testset "infer" begin
    @testset "infer_box() runs" begin
        # very small patch of sky that turns out to have 4 sources.
        # We checked that this patch is in the given field.
        box = ParallelRun.BoundingBox(164.39, 164.41, 39.11, 39.13)
        images = SampleData.get_sdss_images(3900, 6, 269)
        result = ParallelRun.infer_box(images, box; method=:single, do_vi=true)
    end

    @testset "infer_box(..., do_vi=false) runs" begin
        # very small patch of sky that turns out to have 4 sources.
        # We checked that this patch is in the given field.
        box = ParallelRun.BoundingBox(164.39, 164.41, 39.11, 39.13)
        images = SampleData.get_sdss_images(3900, 6, 269)
        result = ParallelRun.infer_box(images, box; method=:single, do_vi=false,
                                       config=Config(2.0, 3, 2))
    end

    @testset "infer_box with directories runs" begin
        box = ParallelRun.BoundingBox("164.39", "164.41", "39.11", "39.13")
        rcfs = [RunCamcolField(3900, 6, 269)]
        datadir = SampleData.DATADIR
        ParallelRun.infer_box(box, datadir, datadir)
    end
end
