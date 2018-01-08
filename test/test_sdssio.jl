using Celeste.SDSSIO
import Celeste.SDSSIO: SDSSBackground, RunCamcolField

function load_from_dataset(dataset)
    rcf = Celeste.RunCamcolField(4263, 5, 119)
    SDSSIO.load_field_images(dataset, rcf)
    SDSSIO.load_field_catalog(dataset, rcf)
end

@testset "sdssio" begin

@testset "sky interpolations" begin
    small_sky = [1.  2.  3.  4.;
                 5.  6.  7.  8.;
                 9. 10. 11. 12]
    sky_x = [0.1, 2.5]
    sky_y = [0.5, 2.5, 4.]
    sky = SDSSBackground(small_sky, sky_x, sky_y, ones(2))

    @test sky[1, 1] ≈ 1.0
    @test sky[2, 1] ≈ 7.0
    @test sky[1, 2] ≈ 2.5
    @test sky[2, 2] ≈ 8.5
    @test sky[1, 3] ≈ 4.0
    @test sky[2, 3] ≈ 10.0
end

@testset "test coordinates out of bounds" begin
    small_sky = [1.  2.  3.  4.;
            5.  6.  7.  8.;
            9. 10. 11. 12]
    sky_x = [-5.0, 4.0]
    sky_y = [-4.0, 5.0]
    sky = SDSSBackground(small_sky, sky_x, sky_y, ones(2))

    @test sky[1,1] == 1.0
    @test sky[1,2] == 4.0
    @test sky[2,1] == 9.0
    @test sky[2,2] == 12.0
end

@testset "read_photoobj handles missing table extensions" begin
    # get an example of a photoobj file with a missing table
    SampleData.get_sdss_catalog(6597, 4, 25)
end


@testset "SDSSDataSet variations" begin
    rcf = Celeste.RunCamcolField(4263, 5, 119)

    # #nsure relevant data is downloaded (this also tests I/O with
    # SDSSDataSet defaults)
    SampleData.get_sdss_images(4263, 5, 119)
    SampleData.get_sdss_catalog(4263, 5, 119)

    # test with slurp = true
    dataset = SDSSDataSet("data"; slurp = true)
    SDSSIO.load_field_images(dataset, rcf)
    SDSSIO.load_field_catalog(dataset, rcf)

    # test with SDSS layout
    rm("data/dr12", force = true, recursive=true)
    cd("data") do
        # Create the original SDSS directory layout for testing purposes
        mkpath("dr12/boss/photo/redux/301/4263/objcs/5")
        mkpath("dr12/boss/photoObj/frames/301/4263/5")
        mkpath("dr12/boss/photoObj/301/4263/5")
        for band in ['z','g','r','i','u']
            cp("4263/5/119/frame-$band-004263-5-0119.fits", "dr12/boss/photoObj/frames/301/4263/5/frame-$band-004263-5-0119.fits")
            cp("4263/5/119/fpM-004263-$(band)5-0119.fit", "dr12/boss/photo/redux/301/4263/objcs/5/fpM-004263-$(band)5-0119.fit")
        end
        cp("4263/5/119/psField-004263-5-0119.fit", "dr12/boss/photo/redux/301/4263/objcs/5/psField-004263-5-0119.fit")
        cp("4263/5/119/photoObj-004263-5-0119.fits", "dr12/boss/photoObj/301/4263/5/photoObj-004263-5-0119.fits")
        cp("4263/5/photoField-004263-5.fits", "dr12/boss/photoObj/301/4263/photoField-004263-5.fits")
    end

    dataset = SDSSDataSet("data/dr12"; dirlayout = :sdss)
    SDSSIO.load_field_images(dataset, rcf)
    SDSSIO.load_field_catalog(dataset, rcf)

    rm("data/dr12", recursive=true)

    worker = addprocs(1)[]
    fetch(@spawnat worker Base.require(:Celeste))

    # Test fetching over the network The basedir is relative, if the
    # worker doesn't try to fetch via the network after this, it'll
    # fail
    fetch(@spawnat worker cd("/tmp"))
    dataset = SDSSDataSet("data"; iostrategy = :masterrpc)
    remotecall_fetch(SDSSIO.load_field_images, worker, dataset, rcf)
    remotecall_fetch(SDSSIO.load_field_catalog, worker, dataset, rcf)
    rmprocs([worker])
end

end
