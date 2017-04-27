using Base.Test
import SensitiveFloats: zero_sensitive_float_array


@testset "sky noise estimates" begin
    images_vec = Vector{Vector{Image}}(2)
    ea, vp, three_bodies = gen_three_body_dataset()  # synthetic
    images_vec[1] = ea.images
    images_vec[2] = SampleData.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf")  # real

    for images in images_vec
        for b in 1:5
            img = images[b]
            epsilon = img.sky[div(img.H, 2), div(img.W, 2)]
            sdss_sky_estimate = epsilon * median(img.iota_vec)
            crude_estimate = median(img.pixels)
            @test isapprox(sdss_sky_estimate / crude_estimate, 1.0, atol=0.3)
        end
    end
end


@testset "zero sensitive float array" begin
    S = 3
    sf_vec = zero_sensitive_float_array(Float64, length(ids), S, 3, 5)
    @test size(sf_vec) == (3, 5)
    for sf in sf_vec
        @test sf.v[] == 0
        @test all([ all(sf_d .== 0) for sf_d in sf.d ])
        @test all(sf.h .== 0)
    end

    # Test that the sensitive floats are distinct and not pointers to the
    # same sensitive float.
    sf_vec[1].v[] = rand()
    sf_vec[1].d[:, 1] = rand(length(CanonicalParams))
    sf_vec[1].h[:, :] = 2 * diagm(ones(size(sf_vec[1].h, 1)))

    for ind in 2:length(sf_vec)
        sf = sf_vec[ind]
        @test sf.v[] == 0
        @test all([ all(sf_d .== 0) for sf_d in sf.d ])
        @test all(sf.h .== 0)
    end
end
