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

    @testset "box_from_catalog: check box is not too big" begin
        wd = pwd()
        cd(datadir)
        run(`make RUN=4114 CAMCOL=3 FIELD=127`)
        run(`make RUN=4114 CAMCOL=4 FIELD=127`)
        cd(wd)

        rcfs = [RunCamcolField(4114, 3, 127), RunCamcolField(4114, 4, 127)]
        strategy = PlainFITSStrategy(datadir)
        images = SDSSIO.load_field_images(strategy, rcfs)
        catalog = SDSSIO.read_photoobj_files(strategy, rcfs)

        patches = [ImagePatch(img, box_from_catalog(img, entry))
                   for entry in catalog, img in images]

        # star at RA, Dec = (309.49754066435867, 45.54976572870953)
        entry_id = 429

        neighbors = find_neighbors(patches, entry_id)

        # there's a lot near this star, but not a lot that overlaps with it, see
        # http://skyserver.sdss.org/dr10/en/tools/explore/summary.aspx?id=0x112d1012607f050a
        @test length(neighbors) < 5
    end
end
