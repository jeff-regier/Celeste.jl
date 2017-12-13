import Celeste: detect_sources
import Celeste.Coordinates: match_coordinates
using Celeste.SDSSIO

@testset "detection" begin

# Compare the initial catalog produced by detect_sources() versus the
# SDSS primary catalog.
@testset "detect_sources() vs SDSS catalog" begin
    images = SampleData.get_sdss_images(4263, 5, 119)
    sdss_catalog = SampleData.get_sdss_catalog(4263, 5, 119)

    catalog, _ = detect_sources(images)

    ra = [ce.pos[1] for ce in catalog]
    dec = [ce.pos[2] for ce in catalog]
    sdss_ra = [ce.pos[1] for ce in sdss_catalog]
    sdss_dec = [ce.pos[2] for ce in sdss_catalog]
    idx, dists = match_coordinates(ra, dec, sdss_ra, sdss_dec)

    # Test that there are a bunch of coordinates that match within 0.5 arcsec
    # (This is a basic sanity check, not a very strict test.)
    @test sum(dists .< 0.5/3600.) > 600
end

end
