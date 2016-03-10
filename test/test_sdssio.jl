import Celeste.SDSSIO

function test_interp_sky()
    data = [1.  2.  3.  4.;
            5.  6.  7.  8.;
            9. 10. 11. 12]
    xcoords = [0.1, 2.5]
    ycoords = [0.5, 2.5, 4.]
    result = SDSSIO.interp_sky(data, xcoords, ycoords)
    @test size(result) == (2, 3)
    @test_approx_eq result[1, 1] 1.0
    @test_approx_eq result[2, 1] 7.0
    @test_approx_eq result[1, 2] 2.5
    @test_approx_eq result[2, 2] 8.5
    @test_approx_eq result[1, 3] 4.0
    @test_approx_eq result[2, 3] 10.0
end

test_interp_sky()
