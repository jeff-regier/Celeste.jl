using Celeste
using CelesteTypes
using Base.Test
using Distributions
using SampleData
using Transform

import Synthetic
import WCS

import GSL.deriv_central

println("Running derivative tests.")


# verify derivatives of fun_to_test by finite differences for functions of
# ModelParams that return a SensitiveFloat.
function test_by_finite_differences(fun_to_test::Function, mp::ModelParams)
    f::SensitiveFloat = fun_to_test(mp)

    for s in 1:mp.S
        alignment = align(f.ids, CanonicalParams)
        for p1 in 1:length(alignment)
            p0 = alignment[p1]

            fun_to_test_2(epsilon::Float64) = begin
                vp_local = deepcopy(mp.vp)
                vp_local[s][p0] += epsilon
                mp_local = deepcopy(mp)
                mp_local.vp = vp_local
                f_local::SensitiveFloat = fun_to_test(mp_local)
                f_local.v
            end

            numeric_deriv, abs_err = deriv_central(fun_to_test_2, 0., 1e-3)
            info(string("deriv #$p0 (s: $s): $numeric_deriv vs $(f.d[p1, s]) ",
                        "[tol: $abs_err]"))
            obs_err = abs(numeric_deriv - f.d[p1, s])
            @test(obs_err < 1e-11 ||
                  abs_err < 1e-4 ||
                  abs_err / abs(numeric_deriv) < 1e-4)
            @test_approx_eq_eps numeric_deriv f.d[p1, s] 10abs_err
        end
    end
end


# verify derivatives of fun_to_test by finite differences for functions of
# numeric vectors that return a tuple (value, gradient).
function test_by_finite_differences(fun_to_test::Function, x::Vector{Float64})
    value, grad = fun_to_test(x)

    for s in 1:length(x)
        fun_to_test_2(epsilon::Float64) = begin
            x_local = deepcopy(x)
            x_local[s] += epsilon
            f_local = fun_to_test(x_local)
            f_local[1]
        end

        numeric_deriv, abs_err = deriv_central(fun_to_test_2, 0., 1e-3)
        info("deriv #$s: $numeric_deriv vs $(grad[s]) [tol: $abs_err]")
        obs_err = abs(numeric_deriv - grad[s])
        @test(obs_err < 1e-11 ||
              abs_err < 1e-4 ||
              abs_err / abs(numeric_deriv) < 1e-4)
        @test_approx_eq_eps numeric_deriv grad[s] 10abs_err
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
    blob, mp, body, tiled_blob = gen_sample_galaxy_dataset();

    function wrap_star(mmp)
        m_pos = Float64[9, 10.]
        wcs_jacobian = WCS.pixel_world_jacobian(blob[3].wcs, m_pos)
        #mmp = ModelInit.initialize_model_params(blob, tiled_blob, body)
        star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mmp, 3)
        fs0m = zero_sensitive_float(StarPosParams)
        ElboDeriv.accum_star_pos!(star_mcs[1,1], m_pos, fs0m, wcs_jacobian)
        fs0m
    end
    test_by_finite_differences(wrap_star, mp)

    function wrap_galaxy(mmp)
        m_pos = Float64[9, 10]
        wcs_jacobian = WCS.pixel_world_jacobian(blob[3].wcs, m_pos)
        #mmp = ModelInit.initialize_model_params(blob, tiled_blob, body)
        star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mmp, 3)
        fs1m = zero_sensitive_float(GalaxyPosParams)
        ElboDeriv.accum_galaxy_pos!(gal_mcs[1,1,1,1], m_pos, fs1m, wcs_jacobian)
        fs1m
    end
    test_by_finite_differences(wrap_galaxy, mp)
end


function test_accum_pixel_source_derivs()
    blob, mp0, body, tiled_blob = gen_sample_galaxy_dataset();

    function wrap_apss_ef(mmp)
        m_pos = [9, 10.]
        wcs_jacobian = WCS.pixel_world_jacobian(blob[1].wcs, m_pos)
        star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mmp, 3)
        fs0m = zero_sensitive_float(StarPosParams)
        fs1m = zero_sensitive_float(GalaxyPosParams)
        E_G = zero_sensitive_float(CanonicalParams)
        var_G = zero_sensitive_float(CanonicalParams)
        sb = ElboDeriv.SourceBrightness(mmp.vp[1])
        ElboDeriv.accum_pixel_source_stats!(sb, star_mcs, gal_mcs,
            mmp.vp[1], 1, 1, m_pos, 1, fs0m, fs1m, E_G, var_G, wcs_jacobian)
        E_G
    end
    test_by_finite_differences(wrap_apss_ef, mp0)

    function wrap_apss_varf(mmp)
        star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mmp, 3)
        fs0m = zero_sensitive_float(StarPosParams)
        fs1m = zero_sensitive_float(GalaxyPosParams)
        E_G = zero_sensitive_float(CanonicalParams)
        var_G = zero_sensitive_float(CanonicalParams)
        sb = ElboDeriv.SourceBrightness(mmp.vp[1])
        m_pos = [9, 10.]
        wcs_jacobian = WCS.pixel_world_jacobian(blob[3].wcs, m_pos)
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
    blob, mp0, body, tiled_blob = gen_sample_galaxy_dataset();

    function wrap_likelihood_b1(mmp)
        ElboDeriv.elbo_likelihood(fill(tiled_blob[1], 1), mmp)
    end
    test_by_finite_differences(wrap_likelihood_b1, mp0)

    function wrap_likelihood_b5(mmp)
        ElboDeriv.elbo_likelihood(fill(tiled_blob[5], 1), mmp)
    end
    test_by_finite_differences(wrap_likelihood_b5, mp0)

    function wrap_elbo(mmp)
        ElboDeriv.elbo(tiled_blob, mmp)
    end
    test_by_finite_differences(wrap_elbo, mp0)
end


function test_elbo_derivs_with_transform()
    blob, mp0, body, tiled_blob = gen_sample_galaxy_dataset();
    trans = get_mp_transform(mp0, loc_width=1.0);

    omitted_ids = Int64[];
    x0 = trans.vp_to_array(mp0.vp, omitted_ids)

    # f is a function of a ModelParams object that returns a SensitiveFloat.
    function wrap_function(f::Function)
        function wrapped_f(x)
            mmp = deepcopy(mp0)
            trans.array_to_vp!(x, mmp.vp, omitted_ids)
            result = f(mmp)
            result_trans = trans.transform_sensitive_float(result, mmp)
            result_trans.v, reduce(vcat, result_trans.d)
        end
        wrapped_f
    end

    wrap_likelihood_b1 =
      wrap_function(mmp -> ElboDeriv.elbo_likelihood(fill(tiled_blob[1], 1), mmp))
    test_by_finite_differences(wrap_likelihood_b1, x0)

    wrap_likelihood_b5 =
      wrap_function(mmp -> ElboDeriv.elbo_likelihood(fill(tiled_blob[5], 1), mmp))
    test_by_finite_differences(wrap_likelihood_b5, x0)

    wrap_elbo = wrap_function(mmp -> ElboDeriv.elbo(tiled_blob, mmp))
    test_by_finite_differences(wrap_elbo, x0)
end


function test_derivative_transform()
  box_param = [1.0, 2.0, 1.001]
  lower_bounds = [-1.0, -2.0, 1.0]
  upper_bounds = [2.0, Inf, Inf]
  scales = [ 2.0, 3.0, 100.0 ]
  N = length(box_param)

  function get_free_param(box_param, lower_bounds, upper_bounds, scales)
    free_param = zeros(Float64, N)
    for n=1:N
      free_param[n] =
        Transform.unbox_parameter(
          box_param[n], lower_bounds[n], upper_bounds[n], scales[n])
    end
    free_param
  end

  function get_box_param(free_param, lower_bounds, upper_bounds, scales)
    box_param = zeros(Float64, N)
    for n=1:N
      box_param[n] =
        Transform.box_parameter(
          free_param[n], lower_bounds[n], upper_bounds[n], scales[n])
    end
    box_param
  end

  # Return value, gradient
  function box_function(box_param)
    sum(box_param .^ 2), 2 * box_param
  end

  function free_function(free_param)
    box_param = get_box_param(free_param, lower_bounds, upper_bounds, scales)
    v, box_deriv = box_function(box_param)
    free_deriv = zeros(Float64, N)
    for n=1:N
      free_deriv[n] = Transform.unbox_derivative(
        box_param[n], box_deriv[n], lower_bounds[n], upper_bounds[n], scales[n])
    end
    v, free_deriv
  end

  free_param = get_free_param(box_param, lower_bounds, upper_bounds, scales)
  test_by_finite_differences(free_function, free_param)
end

test_kl_divergence_derivs()
test_brightness_derivs()
test_accum_pixel_source_derivs()
test_elbo_derivs()
test_derivative_transform()
test_elbo_derivs_with_transform()
