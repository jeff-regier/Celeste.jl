import GSL.deriv_central
using Constrain


# verify derivatives of fun_to_test by finite differences
function test_by_finite_differences(fun_to_test::Function, mp::ModelParams,
                                    trans::DataTransform)

    f::SensitiveFloat = fun_to_test(mp)
    f_trans = trans.transform_sensitive_float(f, mp)
    vp_trans = trans.from_vp(mp.vp)

    for s in 1:mp.S
        for p1 in 1:length(f_trans.param_index)
            p0 = f_trans.param_index[p1]

            fun_to_test_2(epsilon::Float64) = begin
                # Perturb in the transformed space.
                vp_trans_local = deepcopy(vp_trans)
                vp_trans_local[s][p0] += epsilon
                vp_local = trans.to_vp(vp_trans_local)
                mp_local = ModelParams(vp_local, mp.pp, mp.patches, mp.tile_width)
                f_local::SensitiveFloat = fun_to_test(mp_local)
                f_local.v
            end

            numeric_deriv, abs_err = deriv_central(fun_to_test_2, 0., 1e-3)
            info("deriv #$p0 (s: $s): $numeric_deriv vs $(f_trans.d[p1, s]) [tol: $abs_err]")
            obs_err = abs(numeric_deriv - f_trans.d[p1, s]) 
            @test obs_err < 1e-11 || abs_err < 1e-4 || abs_err / abs(numeric_deriv) < 1e-4
            @test_approx_eq_eps numeric_deriv f_trans.d[p1, s] 10abs_err
        end
    end
end


function test_by_finite_differences(fun_to_test::Function, mp::ModelParams)
    # By default, test with the identity transform.
    test_by_finite_differences(fun_to_test, mp, identity_transform)
end


function test_brightness_derivs()
    blob, mp0, three_bodies = gen_three_body_dataset()

    for i = 1:Ia
        for b = [3,4,2,5,1]
            function wrap_source_brightness(mp)
                sb = ElboDeriv.SourceBrightness(mp.vp[1])
                ret = zero_sensitive_float([1,2,3], all_params)
                ret.v = sb.E_l_a[b, i].v
                ret.d[:, 1] = sb.E_l_a[b, i].d
                ret
            end
            test_by_finite_differences(wrap_source_brightness, mp0)

            function wrap_source_brightness_3(mp)
                sb = ElboDeriv.SourceBrightness(mp.vp[1])
                ret = zero_sensitive_float([1,2,3], all_params)
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
        star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[3].psf, mmp)
        fs0m = zero_sensitive_float([1], star_pos_params)
        ElboDeriv.accum_star_pos!(star_mcs[1,1], [9, 10.], fs0m)
        fs0m
    end
    test_by_finite_differences(wrap_star, mp)

    function wrap_galaxy(mmp)
        star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[3].psf, mmp)
        fs1m = zero_sensitive_float([1], galaxy_pos_params)
        ElboDeriv.accum_galaxy_pos!(gal_mcs[1,1,1,1], [9, 10.], fs1m)
        fs1m
    end
    test_by_finite_differences(wrap_galaxy, mp)
end


function test_accum_pixel_source_derivs()
    blob, mp0, body = gen_sample_galaxy_dataset()

    function wrap_apss_ef(mmp)
        star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[1].psf, mmp)
        fs0m = zero_sensitive_float([1], star_pos_params)
        fs1m = zero_sensitive_float([1], galaxy_pos_params)
        E_G = zero_sensitive_float([1], all_params)
        var_G = zero_sensitive_float([1], all_params)
        sb = ElboDeriv.SourceBrightness(mmp.vp[1])
        m_pos = [9, 10.]
        ElboDeriv.accum_pixel_source_stats!(sb, star_mcs, gal_mcs,
            mmp.vp[1], 1, 1, m_pos, 1, fs0m, fs1m, E_G, var_G)
        E_G
    end
    test_by_finite_differences(wrap_apss_ef, mp0)

    function wrap_apss_varf(mmp)
        star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[3].psf, mmp)
        fs0m = zero_sensitive_float([1], star_pos_params)
        fs1m = zero_sensitive_float([1], galaxy_pos_params)
        E_G = zero_sensitive_float([1], all_params)
        var_G = zero_sensitive_float([1], all_params)
        sb = ElboDeriv.SourceBrightness(mmp.vp[1])
        m_pos = [9, 10.]
        ElboDeriv.accum_pixel_source_stats!(sb, star_mcs, gal_mcs,
            mmp.vp[1], 1, 1, m_pos, 3, fs0m, fs1m, E_G, var_G)
        var_G
    end
    test_by_finite_differences(wrap_apss_varf, mp0)

end


function test_kl_divergence_derivs()
    blob, mp0, three_bodies = gen_three_body_dataset()

    function wrap_kl_a(mp)
        accum = zero_sensitive_float([1:3], all_params)
        ElboDeriv.subtract_kl_a!(1, mp, accum)
        accum
    end
    test_by_finite_differences(wrap_kl_a, mp0)

    function wrap_kl_r(mp)
        accum = zero_sensitive_float([1:3], all_params)
        ElboDeriv.subtract_kl_r!(1, 1, mp, accum)
        accum
    end
    test_by_finite_differences(wrap_kl_r, mp0)

    function wrap_kl_k(mp)
        accum = zero_sensitive_float([1:3], all_params)
        ElboDeriv.subtract_kl_k!(1, 1, mp, accum)
        accum
    end
    test_by_finite_differences(wrap_kl_k, mp0)

    function wrap_kl_c(mp)
        accum = zero_sensitive_float([1:3], all_params)
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


test_kl_divergence_derivs()
test_brightness_derivs()
test_accum_pos_derivs()
test_accum_pixel_source_derivs()
test_elbo_derivs()

