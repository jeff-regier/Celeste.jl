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


function test_source_division_parallelism()
    box = ParallelRun.BoundingBox(164.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]
    results = ParallelRun.divide_sources_and_infer(box, datadir)
    @test length(results) == 4
end


function test_load_active_pixels()
    images, ea, one_body = gen_sample_star_dataset()

    ea.active_pixels = ActivePixel[]
    Infer.load_active_pixels!(ea; min_radius_pix=0.0)

   # these images have 20 * 23 * 5 = 2300 pixels in total.
   # the star is bright but it doesn't cover the whole image.
   # it's hard to say exactly how many pixels should be active,
   # but not all of them, and not none of them.
   @test 100 < length(ea.active_pixels) < 2000
end


test_load_active_pixels()
test_source_division_parallelism()
test_infer_single()
