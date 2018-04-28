import Celeste: BoundingBox
import Celeste.DECALSIO: DECALSDataSet, _get_overlapping_ccds


@testset "decalsio" begin

    # ensure files downloaded
    DATADIR = joinpath(Pkg.dir("Celeste"), "test", "data", "decam")
    wd = pwd()
    cd(DATADIR)
    make_output = readstring(`make survey-ccds-dr5-0.0-0.5-0.0-0.5.fits`)
    if !startswith(make_output, "make: Nothing to be done for")
        print(make_output)  # only print output if something was done
    end
    cd(wd)


    @testset "overlapping ccds" begin
        dataset = DECALSDataSet(DATADIR,
                                "survey-ccds-dr5-0.0-0.5-0.0-0.5.fits")
        box = BoundingBox(0.0, 0.5, 0.0, 0.5)
        idx = _get_overlapping_ccds(dataset, box)
        @test length(idx) == 134  # all CCDs in FITS file should overlap.

        small_box = BoundingBox(0.2, 0.4, 0.2, 0.4)
        idx2 = _get_overlapping_ccds(dataset, small_box)
        @test length(idx2) == 82  # determined from running code!
                                  # (But should be less than 134)
    end
end
