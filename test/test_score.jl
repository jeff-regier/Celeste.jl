## test the main entry point in Celeste: the `infer` function
import Celeste
import JLD

"""
test infer with a single (run, camcol, field).
This is basically just to make sure it runs at all.
"""
function test_infer_single()
    # very small patch of sky that turns out to have 4 sources.
    # We checked that this patch is in the given field.
    ra_range = (0.5, 0.52)
    dec_range = (0.5, 0.52)
    fieldids = [(4263, 5, 119)]

    inferences = Celeste.infer(ra_range, dec_range, fieldids, [datadir])

    coadd_path = joinpath(datadir, "coadd_test_catalog.fit")
    Celeste.score(ra_range, dec_range, inferences, coadd_path)
end

test_infer_single()
