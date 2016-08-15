## test the main entry point in Celeste: the `infer` function
import JLD


"""
test infer with a single (run, camcol, field).
This is basically just to make sure it runs at all.
"""
function test_infer_single()
    # very small patch of sky that turns out to have 4 sources.
    # We checked that this patch is in the given field.
    box = Celeste.BoundingBox(164.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]
    result = Celeste.one_node_infer(field_triplets, datadir; box=box)
end


function test_source_division_parallelism()
    box = Celeste.BoundingBox(164.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]
    results = Celeste.divide_sources_and_infer(box, datadir)
    @test length(results) == 4
end


test_source_division_parallelism()
test_infer_single()
