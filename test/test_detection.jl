import Celeste: detect_sources
import Celeste.Coordinates: match_coordinates
using Celeste.SDSSIO

@testset "detection" begin

# Compare the initial catalog produced by detect_sources() versus the
# SDSS primary catalog.
@testset "detect_sources() vs SDSS catalog" begin
    rcf = RunCamcolField(4263, 5, 119)
    cd(datadir)
    run(`make RUN=$(rcf.run) CAMCOL=$(rcf.camcol) FIELD=$(rcf.field)`)
    cd(wd)
    stagedir = joinpath(datadir, string(rcf.run), string(rcf.camcol),
                        string(rcf.field))

    # SDSS catalog
    fname_photoobj = joinpath(stagedir, SDSSIO.filename(SDSSIO.PhotoObj(rcf)))
    sdss_catalog = SDSSIO.read_photoobj(FITSIO.FITS(fname_photoobj))

    # Get raw images
    strategy = SDSSIO.PlainFITSStrategy(datadir)
    images = SDSSIO.load_field_images(strategy, rcf)
    catalog, source_radii = detect_sources(images)

    ra = [ce.pos[1] for ce in catalog]
    dec = [ce.pos[2] for ce in catalog]
    idx, dists = match_coordinates(ra, dec,
                                   sdss_catalog["ra"], sdss_catalog["dec"])

    # Test that there are a bunch of coordinates that match within 0.5 arcsec
    # (This is a basic sanity check, not a very strict test.)
    @test sum(dists .< 0.5/3600.) > 600
end

end
