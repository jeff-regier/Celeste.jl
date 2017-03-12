import Celeste.SDSSIO: SkyIntensity

function test_interp_sky()
    small_sky = [1.  2.  3.  4.;
            5.  6.  7.  8.;
            9. 10. 11. 12]
    sky_x = [0.1, 2.5]
    sky_y = [0.5, 2.5, 4.]
    sky = SkyIntensity(small_sky, sky_x, sky_y)
    @test sky[1, 1] ≈ 1.0
    @test sky[2, 1] ≈ 7.0
    @test sky[1, 2] ≈ 2.5
    @test sky[2, 2] ≈ 8.5
    @test sky[1, 3] ≈ 4.0
    @test sky[2, 3] ≈ 10.0
end

# test coordinates out of bounds
function test_interp_sky_oob()
    small_sky = [1.  2.  3.  4.;
            5.  6.  7.  8.;
            9. 10. 11. 12]
    sky_x = [-5.0, 4.0]
    sky_y = [-4.0, 5.0]
    sky = SkyIntensity(small_sky, sky_x, sky_y)

    @test sky[1,1] == 1.0
    @test sky[1,2] == 4.0
    @test sky[2,1] == 9.0
    @test sky[2,2] == 12.0
end

# test that read_photoobj handles missing table extensions.
function test_read_photoobj_missing()

    # get an example of a photoobj file with a missing table
    fname = joinpath(Pkg.dir("Celeste"), "test", "data",
                     "photoObj-006597-4-0025.fits")
    if !isfile(fname)
        run(`curl --create-dirs -o $fname https://data.sdss.org/sas/dr12/boss/photoObj/301/6597/4/photoObj-006597-4-0025.fits`)
    end

    catalog = SDSSIO.read_photoobj(fname)
end


test_read_photoobj_missing()
test_interp_sky()
test_interp_sky_oob()
