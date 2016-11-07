## test the main entry point in Celeste: the `infer` function
import JLD


"""
test infer with a single (run, camcol, field).
This is basically just to make sure it runs at all.
"""
function test_infer_single()
    # very small patch of sky that turns out to have 4 sources.
    # We checked that this patch is in the given field.
    box = ParallelRun.BoundingBox(164.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]
    result = ParallelRun.one_node_infer(field_triplets, datadir; box=box)
end


function test_infer_rcf()
    resfile = joinpath(datadir, "celeste-003900-6-0269.jld")
    rm(resfile, force=true)

    rcf = RunCamcolField(3900, 6, 269)
    objid = "1237662226208063492"
    ParallelRun.infer_rcf(rcf, datadir, datadir; objid=objid)

    @test isfile(resfile)
    println(filesize(resfile))
    @test filesize(resfile) > 1000  # should be about 15 KB
    rm(resfile)
end


function test_source_division_parallelism()
    box = ParallelRun.BoundingBox(164.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]
    results = ParallelRun.divide_sources_and_infer(box, datadir)
    @test length(results) == 4
end


function test_load_active_pixels()
    images, ea, one_body = gen_sample_star_dataset()

    # these images have 20 * 23 * 5 = 2300 pixels in total.
    # the star is bright but it doesn't cover the whole image.
    # it's hard to say exactly how many pixels should be active,
    # but not all of them, and not none of them.
    ea.active_pixels = ActivePixel[]
    Infer.load_active_pixels!(ea; min_radius_pix=0.0)
    @test 100 < length(ea.active_pixels) < 2000

    # most star light (>90%) should be recorded by the active pixels
    active_photons = 0.0
    for ap in ea.active_pixels
        tile = ea.images[ap.n].tiles[ap.tile_ind]
        active_photons += tile.pixels[ap.h, ap.w] - tile.epsilon_mat[ap.h, ap.w]
    end

    total_photons = 0.0
    for img in ea.images
        for t in img.tiles
            total_photons += sum(t.pixels) - sum(t.epsilon_mat)
        end
    end

    @test active_photons <= total_photons  # sanity check
    @test active_photons > 0.9 * total_photons

    # a really dim star never exceeds the background intensity by much
    ea.vp[1][ids.r1] = -999.  # very dim
    ea.active_pixels = ActivePixel[]
    Infer.load_active_pixels!(ea; min_radius_pix=0.0)
    @test length(ea.active_pixels) == 0

    # only 2 pixels per image are within 0.6 pixels of the
    # source's center (10.9, 11.5)
    ea.active_pixels = ActivePixel[]
    Infer.load_active_pixels!(ea; min_radius_pix=0.6)
    @test length(ea.active_pixels) == 2 * 5
end


test_source_division_parallelism()
test_infer_single()
test_infer_rcf()
test_load_active_pixels()
