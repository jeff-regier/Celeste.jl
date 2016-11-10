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
    Infer.load_active_pixels!(ea; min_radius_pix=0.0)

    # most star light (>90%) should be recorded by the active pixels
    num_active_photons = 0.0
    num_active_pixels = 0
    for n in 1:ea.N
        img = ea.images[n]
        p = ea.patches[1, n]
        H2, W2 = size(p.active_pixel_bitmap)
        for w2 in 1:W2, h2 in 1:H2
            # (h2, w2) index the local patch, while (h, w) index the image
            h = p.bitmap_corner[1] + h2
            w = p.bitmap_corner[2] + w2
            num_active_photons += img.pixels[h, w] - img.epsilon_mat[h, w]
            num_active_pixels += 1
        end
    end

    @test 100 < num_active_pixels < 2200

    total_photons = 0.0
    for img in ea.images
        total_photons += sum(img.pixels) - sum(img.epsilon_mat)
    end

    @test num_active_photons <= total_photons  # sanity check
    @test num_active_photons > 0.9 * total_photons

    # super dim images
    for img in images
        img.pixels[:,:] = img.epsilon_mat[:,:]
    end

    # only 2 pixels per image are within 0.6 pixels of the
    # source's center (10.9, 11.5)
    Infer.load_active_pixels!(ea; min_radius_pix=0.6)

    for n in 1:ea.N
#  FIXME: is load active pixels off by (0.5, 0.5)?
#        @test sum(ea.patches[1,n].active_pixel_bitmap) == 2
    end
end


test_load_active_pixels()
test_source_division_parallelism()
test_infer_single()
test_infer_rcf()
