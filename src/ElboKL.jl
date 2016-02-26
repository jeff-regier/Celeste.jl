import ForwardDiff
import KL


@doc """
Subtract the KL divergence from the prior for c
""" ->
function subtract_kl_c{NumType <: Number}(
    d::Int64, i::Int64, vs::Vector{NumType}, pp::PriorParams)

  a = vs[ids.a[i]]
  k = vs[ids.k[d, i]]

  pp_kl_cid = KL.gen_diagmvn_mvn_kl(pp.c_mean[:, d, i], pp.c_cov[:, :, d, i])
  # (v, (d_c1, d_c2)) = pp_kl_cid(vs[ids.c1[:, i]], vs[ids.c2[:, i]])
  # accum.v[1] -= v * a * k
  -pp_kl_cid(vs[ids.c1[:, i]], vs[ids.c2[:, i]]) * a * k

  # accum.d[ids.k[d, i], s] -= a * v
  # accum.d[ids.c1[:, i], s] -= a * k * d_c1
  # accum.d[ids.c2[:, i], s] -= a * k * d_c2
  # accum.d[ids.a[i], s] -= k * v
end

@doc """
Subtract the KL divergence from the prior for k
""" ->
function subtract_kl_k{NumType <: Number}(
  i::Int64, vs::Vector{NumType}, pp::PriorParams)

    pp_kl_ki = KL.gen_categorical_kl(pp.k[:, i])
    # (v, (d_k,)) = pp_kl_ki(vs[ids.k[:, i]])
    # accum.v[1] -= v * vs[ids.a[i]]
    # accum.d[ids.k[:, i], s] -= d_k .* vs[ids.a[i]]
    # accum.d[ids.a[i], s] -= v

    -vs[ids.a[i]] * pp_kl_ki(vs[ids.k[:, i]])
end


@doc """
Subtract the KL divergence from the prior for r for object type i.
""" ->
function subtract_kl_r{NumType <: Number}(
  i::Int64, vs::Vector{NumType}, pp::PriorParams)

    a = vs[ids.a[i]]

    pp_kl_r = KL.gen_normal_kl(pp.r_mean[i], pp.r_var[i])
    # (v, (d_r1, d_r2)) = pp_kl_r(vs[ids.r1[i]], vs[ids.r2[i]])

    # accum.v[1] -= v * a
    # accum.d[ids.r1[i], s] -= d_r1 .* a
    # accum.d[ids.r2[i], s] -= d_r2 .* a
    # accum.d[ids.a[i], s] -= v

    v = pp_kl_r(vs[ids.r1[i]], vs[ids.r2[i]])

    -a * v
end


@doc """
Subtract the KL divergence from the prior for a
""" ->
function subtract_kl_a{NumType <: Number}(vs::Vector{NumType}, pp::PriorParams)
    pp_kl_a = KL.gen_categorical_kl(pp.a)
    # (v, (d_a,)) = pp_kl_a(vs[ids.a])
    # accum.v[1] -= v
    # accum.d[ids.a, s] -= d_a
    -pp_kl_a(vs[ids.a])
end


function subtract_kl!{NumType <: Number}(
    mp::ModelParams{NumType}, accum::SensitiveFloat{CanonicalParams, NumType};
    calculate_derivs::Bool=true, calculate_hessian::Bool=true)

  # The KL divergence as a function of the active source variational parameters.
  function subtract_kl_value_wrapper{NumType <: Number}(vp_vec::Vector{NumType})
    elbo_val = zero(NumType)
    vp_active = reshape(vp_vec, length(CanonicalParams), length(mp.active_sources))
    for sa in 1:length(mp.active_sources)
      vs = vp_active[:, sa]
      elbo_val += subtract_kl_a(vs, mp.pp)

      for i in 1:Ia
          elbo_val += subtract_kl_r(i, vs, mp.pp)
          elbo_val += subtract_kl_k(i, vs, mp.pp)
          for d in 1:D
              elbo_val += subtract_kl_c(d, i, vs, mp.pp)
          end
      end
    end
    elbo_val
  end


  vp_vec = reduce(vcat, Vector{NumType}[ mp.vp[sa] for sa in mp.active_sources ])

  const P = length(CanonicalParams)
  Sa = length(mp.active_sources)

  if calculate_derivs
    if calculate_hessian
      hess, all_results =
        ForwardDiff.hessian(subtract_kl_value_wrapper, vp_vec, ForwardDiff.AllResults)
      accum.h += hess
      accum.d += reshape(ForwardDiff.gradient(all_results), P, Sa);
      accum.v[1] += ForwardDiff.value(all_results)
    else
      grad, all_results =
        ForwardDiff.gradient(subtract_kl_value_wrapper, vp_vec, ForwardDiff.AllResults)
      accum.d += reshape(grad, P, Sa);
      accum.v[1] += ForwardDiff.value(all_results)
    end
  else
    accum.v[1] += subtract_kl_value_wrapper(vp_vec)
  end
end
