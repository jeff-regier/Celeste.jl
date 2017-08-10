
import Celeste.Coordinates: angular_separation, match_coordinates

@testset "coordinates" begin

    @test angular_separation(180., 45., 0., -45.) == 180.
    @test angular_separation(0., 0., 0., 1.) == 1.
    
    ra1 = [2., 4., 5.]
    dec1 = [70., -80., 90.]
    ra2 = [5., 2., 0., 50.]
    dec2 = [89.9, 70.1, 0., -90.]
    idxs, dists = match_coordinates(ra1, dec1, ra2, dec2)

    @test length(idxs) == length(ra1)
    @test idxs[1] == 2 && dists[1] ≈ 0.1
    @test idxs[2] == 4 && dists[2] ≈ 10.
    @test idxs[3] == 1 && dists[3] ≈ 0.1
end
