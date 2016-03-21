import Celeste


function test_score()
    # very small patch of sky that turns out to have 4 sources.
    # We checked that this patch is in the given field.
    ra_range = (0.5, 0.52)
    dec_range = (0.5, 0.52)
    fieldids = [(4263, 5, 119)]

    inferences = Celeste.infer(ra_range, dec_range, fieldids, [datadir])

    coadd_path = joinpath(datadir, "coadd_test_catalog.fit")
    Celeste.score(ra_range, dec_range, inferences, coadd_path)
end

test_score()
