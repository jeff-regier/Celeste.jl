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
    fieldids = [(3900, 6, 269)]
    dirs = [datadir]

    result = Celeste.infer(fieldids, dirs; box=box)
end

test_infer_single()
