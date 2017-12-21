## test the main entry point in Celeste: the `infer` function
import JLD

import Celeste: Config
using Celeste.SDSSIO
using Celeste.ParallelRun
import Celeste.ParallelRun: OptimizedSource
using Celeste.DeterministicVI

function compute_obj_value(images::Vector{<:Image},
                           catalog::Vector{CatalogEntry},
                           box::BoundingBox,
                           results::Vector{OptimizedSource})

    # TODO: This stuff is duplicated from ParallelRun.infer_box.
    # We should refactor infer_box to return the objective value in some way!
    patches = Model.get_sky_patches(images, catalog)
    entry_in_range = entry->((box.ramin < entry.pos[1] < box.ramax) &&
                             (box.decmin < entry.pos[2] < box.decmax))
    target_ids = find(entry_in_range, catalog)

    # There must be a vp for every patch in the call to elbo().
    # So, here we must limit patches to just the targets we optimized
    # and pass [1, 2, 3, ...] as the target indexes.
    ea = ElboArgs(images, patches[target_ids, :],
                  collect(1:length(target_ids)); include_kl=false)
    vp = [r.vs for r in results]
    DeterministicVI.elbo(ea, vp).v[]
end

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

    @testset "joint vs single on overlapping sources" begin
            images = SampleData.get_sdss_images(4263, 5, 119)
        catalog = SampleData.get_sdss_catalog(4263, 5, 119)

        # This box has 3 overlapping objects in it.
        box = BoundingBox(0.467582, 0.473275, 0.588383, 0.595095)

        results_single = ParallelRun.infer_box(images, catalog, box;
                                               method=:single)
        results_joint = ParallelRun.infer_box(images, catalog, box;
                                              method=:joint)

        @test length(results_single) == 3
        @test length(results_joint) == 3

        score_single = compute_obj_value(images, catalog, box, results_single)
        score_joint = compute_obj_value(images, catalog, box, results_joint)

        @test score_joint > score_single
    end
end
