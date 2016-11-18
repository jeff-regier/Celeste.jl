using Base.Test

import Celeste.Model: patch_ctrs_pix, patch_radii_pix
import Celeste.SensitiveFloats.zero_sensitive_float_array


function test_sky_noise_estimates()
    images_vec = Array(Vector{Image}, 2)
    images_vec[1], ea, three_bodies = gen_three_body_dataset()  # synthetic
    images_vec[2] = SampleData.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf")  # real

    for images in images_vec
        for b in 1:5
            sdss_sky_estimate = median(images[b].epsilon_mat) * median(images[b].iota_vec)
            crude_estimate = median(images[b].pixels)
            @test_approx_eq_eps sdss_sky_estimate / crude_estimate 1. .3
        end
    end
end


function test_zero_sensitive_float_array()
    S = 3
    sf_vec = zero_sensitive_float_array(CanonicalParams, Float64, S, 3, 5)
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
    sf_vec[1].h = 2 * diagm(ones(size(sf_vec[1].h, 1)))

    for ind in 2:length(sf_vec)
        sf = sf_vec[ind]
        @test sf.v[] == 0
        @test all([ all(sf_d .== 0) for sf_d in sf.d ])
        @test all(sf.h .== 0)
    end
end


####################################################

test_sky_noise_estimates()
test_zero_sensitive_float_array()
