import Celeste


function test_score()
    # very small patch of sky
    # We checked that this patch is in the given field.
    ra_range = (0.5, 0.52)
    dec_range = (0.5, 0.52)
    fieldid = (4263, 5, 119)

    inferences = Celeste.infer(ra_range, dec_range, [fieldid], [datadir],
                    max_iters=50, ignore_primary_mask=true)

    coadd_path = joinpath(datadir, "coadd_test_catalog.fit")
    Celeste.score(ra_range, dec_range, fieldid,
                    inferences, coadd_path, datadir)
end

test_score()
