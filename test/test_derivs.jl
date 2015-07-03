using Celeste
using CelesteTypes
using Base.Test
using Distributions
using SampleData
import Synthetic
using Transform

import GSL.deriv_central


# verify derivatives of fun_to_test by finite differences
function test_by_finite_differences(fun_to_test::Function, mp::ModelParams)
    f::SensitiveFloat = fun_to_test(mp)

    for s in 1:mp.S
        alignment = align(f.ids, CanonicalParams)
        for p1 in 1:length(alignment)
            p0 = alignment[p1]

            fun_to_test_2(epsilon::Float64) = begin
                vp_local = deepcopy(mp.vp)
                vp_local[s][p0] += epsilon
                mp_local = ModelParams(vp_local, mp.pp, mp.patches, mp.tile_width)
                f_local::SensitiveFloat = fun_to_test(mp_local)
                f_local.v
            end

            numeric_deriv, abs_err = deriv_central(fun_to_test_2, 0., 1e-3)
            info("deriv #$p0 (s: $s): $numeric_deriv vs $(f.d[p1, s]) [tol: $abs_err]")
            obs_err = abs(numeric_deriv - f.d[p1, s]) 
            @test obs_err < 1e-11 || abs_err < 1e-4 || abs_err / abs(numeric_deriv) < 1e-4
            @test_approx_eq_eps numeric_deriv f.d[p1, s] 10abs_err
        end
    end
end


function test_brightness_derivs()
    blob, mp0, three_bodies = gen_three_body_dataset()

    for i = 1:Ia
        for b = [3,4,2,5,1]
            function wrap_source_brightness(mp)
                sb = ElboDeriv.SourceBrightness(mp.vp[1])
                ret = zero_sensitive_float(CanonicalParams, 3)
                ret.v = sb.E_l_a[b, i].v
                ret.d[:, 1] = sb.E_l_a[b, i].d
                ret
            end
            test_by_finite_differences(wrap_source_brightness, mp0)

            function wrap_source_brightness_3(mp)
                sb = ElboDeriv.SourceBrightness(mp.vp[1])
                ret = zero_sensitive_float(CanonicalParams, 3)
                ret.v = sb.E_ll_a[b, i].v
                ret.d[:, 1] = sb.E_ll_a[b, i].d
                ret
            end
            test_by_finite_differences(wrap_source_brightness_3, mp0)
        end
    end
end


function test_accum_pos_derivs()
    blob, mp, body = gen_sample_galaxy_dataset()

    function wrap_star(mmp)
        star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[3].psf, mmp, blob[3].wcs)
        fs0m = zero_sensitive_float(StarPosParams)
        m_pos = Float64[9, 10.]
        wcs_jacobian = pixel_world_jacobian(blob[3].wcs, m_pos)
        ElboDeriv.accum_star_pos!(star_mcs[1,1], m_pos, fs0m, wcs_jacobian)
        fs0m
    end
    test_by_finite_differences(wrap_star, mp)

    function wrap_galaxy(mmp)
        star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[3].psf, mmp, blob[3].wcs)
        fs1m = zero_sensitive_float(GalaxyPosParams)
        m_pos = Float64[9, 10]
        wcs_jacobian = pixel_world_jacobian(blob[3].wcs, m_pos)
        ElboDeriv.accum_galaxy_pos!(gal_mcs[1,1,1,1], m_pos, fs1m, wcs_jacobian)
        fs1m
    end
    test_by_finite_differences(wrap_galaxy, mp)
end


function test_accum_pixel_source_derivs()
    blob, mp0, body = gen_sample_galaxy_dataset()

    function wrap_apss_ef(mmp)
        star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[1].psf, mmp, blob[1].wcs)
        fs0m = zero_sensitive_float(StarPosParams)
        fs1m = zero_sensitive_float(GalaxyPosParams)
        E_G = zero_sensitive_float(CanonicalParams)
        var_G = zero_sensitive_float(CanonicalParams)
        sb = ElboDeriv.SourceBrightness(mmp.vp[1])
        m_pos = [9, 10.]
        wcs_jacobian = pixel_world_jacobian(blob[1].wcs, m_pos)
        ElboDeriv.accum_pixel_source_stats!(sb, star_mcs, gal_mcs,
            mmp.vp[1], 1, 1, m_pos, 1, fs0m, fs1m, E_G, var_G, wcs_jacobian)
        E_G
    end
    test_by_finite_differences(wrap_apss_ef, mp0)

    function wrap_apss_varf(mmp)
        star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[3].psf, mmp, blob[3].wcs)
        fs0m = zero_sensitive_float(StarPosParams)
        fs1m = zero_sensitive_float(GalaxyPosParams)
        E_G = zero_sensitive_float(CanonicalParams)
        var_G = zero_sensitive_float(CanonicalParams)
        sb = ElboDeriv.SourceBrightness(mmp.vp[1])
        m_pos = [9, 10.]
        wcs_jacobian = pixel_world_jacobian(blob[3].wcs, m_pos)
        ElboDeriv.accum_pixel_source_stats!(sb, star_mcs, gal_mcs,
            mmp.vp[1], 1, 1, m_pos, 3, fs0m, fs1m, E_G, var_G, wcs_jacobian)
        var_G
    end
    test_by_finite_differences(wrap_apss_varf, mp0)

end


function test_kl_divergence_derivs()
    blob, mp0, three_bodies = gen_three_body_dataset()

    function wrap_kl_a(mp)
        accum = zero_sensitive_float(CanonicalParams, 3)
        ElboDeriv.subtract_kl_a!(1, mp, accum)
        accum
    end
    test_by_finite_differences(wrap_kl_a, mp0)

    function wrap_kl_r(mp)
        accum = zero_sensitive_float(CanonicalParams, 3)
        ElboDeriv.subtract_kl_r!(1, 1, mp, accum)
        accum
    end
    test_by_finite_differences(wrap_kl_r, mp0)

    function wrap_kl_k(mp)
        accum = zero_sensitive_float(CanonicalParams, 3)
        ElboDeriv.subtract_kl_k!(1, 1, mp, accum)
        accum
    end
    test_by_finite_differences(wrap_kl_k, mp0)

    function wrap_kl_c(mp)
        accum = zero_sensitive_float(CanonicalParams, 3)
        ElboDeriv.subtract_kl_c!(1, 1, 1, mp, accum)
        accum
    end
    test_by_finite_differences(wrap_kl_c, mp0)
end


function test_elbo_derivs()
    blob, mp0, body = gen_sample_galaxy_dataset()

    function wrap_likelihood_b1(mmp)
        ElboDeriv.elbo_likelihood([blob[1]], mmp)
    end
    test_by_finite_differences(wrap_likelihood_b1, mp0)

    function wrap_likelihood_b5(mmp)
        ElboDeriv.elbo_likelihood([blob[5]], mmp)
    end
    test_by_finite_differences(wrap_likelihood_b5, mp0)

    function wrap_elbo(mmp)
        ElboDeriv.elbo([blob], mmp)
    end
    test_by_finite_differences(wrap_elbo, mp0)
end


function test_quadratic_derivatives(trans::DataTransform)
    # A very simple quadratic function to test the derivatives.
    function quadratic_function(mp::ModelParams)
        const centers = collect(linrange(0, 10, length(CanonicalParams)))

        val = zero_sensitive_float(CanonicalParams)
        val.v = sum((mp.vp[1] - centers) .^ 2)
        val.d[:] = 2.0 * (mp.vp[1] - centers)

        val
    end

    # 0.5 is an innocuous value for all parameters.
    mp = empty_model_params(1)
    mp.vp = convert(VariationalParams,[ fill(0.5, length(CanonicalParams)) 
        for s in 1:1 ])
    test_by_finite_differences(quadratic_function, mp)
end

for trans in [ identity_transform, rect_transform, free_transform ]
    test_quadratic_derivatives(trans)
end

test_kl_divergence_derivs()
test_brightness_derivs()
test_accum_pixel_source_derivs()
test_elbo_derivs()

