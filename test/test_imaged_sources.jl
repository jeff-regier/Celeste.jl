using Celeste.Model: boxes_overlap, ImagePatch, box_from_catalog, find_neighbors
using Celeste.SDSSIO
using Base.Test


@testset "ImagePatch" begin
    @testset "boxes_overlap" begin
        @test boxes_overlap((1:0, 1:0), (1:0, 1:0)) == false
        @test boxes_overlap((1:1, 1:0), (1:2, 1:1)) == false
        @test boxes_overlap((1:2, 5:7), (2:3, 3:4)) == false
        @test boxes_overlap((1:2, 5:7), (2:3, 7:10)) == true
    end

    @testset "box_from_catalog: run it and check maxradius" begin
        wd = pwd()
        cd(datadir)
        run(`make RUN=4114 CAMCOL=3 FIELD=127`)
        cd(wd)

        rcfs = [RunCamcolField(4114, 3, 127)]
        strategy = PlainFITSStrategy(datadir)
        images = SDSSIO.load_field_images(strategy, rcfs)
        catalog = SDSSIO.read_photoobj_files(strategy, rcfs)

        patches = [ImagePatch(img, box_from_catalog(img, entry; max_radius=25))
                   for entry in catalog, img in images]

        patch_widths = [length(p.box[1]) for p in patches]

        # check that all obey maxradius
        @test all(patch_widths .<= 51)

        # check that some radii are smaller than 20 (fairly arbitrary number)
        @test sum(patch_widths .<= 40) > length(patch_widths) / 2
    end
end
