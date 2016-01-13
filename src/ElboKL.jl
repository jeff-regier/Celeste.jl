using ForwardDiff


@doc """
Subtract the KL divergence from the prior for c
""" ->
function subtract_kl_c!{NumType <: Number}(
  d::Int64, i::Int64, vs::Vector{NumType}, pp::PriorParams,
  accum::SensitiveFloat{CanonicalParams, NumType})

    a = vs[ids.a[i]]
    k = vs[ids.k[d, i]]

    pp_kl_cid = KL.gen_diagmvn_mvn_kl(pp.c_mean[:, d, i],
                                      pp.c_cov[:, :, d, i])
    (v, (d_c1, d_c2)) = pp_kl_cid(vs[ids.c1[:, i]],
                                        vs[ids.c2[:, i]])
    accum.v -= v * a * k
    # accum.d[ids.k[d, i], s] -= a * v
    # accum.d[ids.c1[:, i], s] -= a * k * d_c1
    # accum.d[ids.c2[:, i], s] -= a * k * d_c2
    # accum.d[ids.a[i], s] -= k * v
end

@doc """
Subtract the KL divergence from the prior for k
""" ->
function subtract_kl_k!{NumType <: Number}(
  i::Int64, vs::Vector{NumType}, pp::PriorParams,
  accum::SensitiveFloat{CanonicalParams, NumType})

    pp_kl_ki = KL.gen_categorical_kl(pp.k[:, i])
    (v, (d_k,)) = pp_kl_ki(mp.vp[s][ids.k[:, i]])
    accum.v -= v * vs[ids.a[i]]
    # accum.d[ids.k[:, i], s] -= d_k .* vs[ids.a[i]]
    # accum.d[ids.a[i], s] -= v
end


@doc """
Subtract the KL divergence from the prior for r for object type i.
""" ->
function subtract_kl_r!{NumType <: Number}(
  i::Int64, vs::Vector{NumType}, pp::PriorParams,
  accum::SensitiveFloat{CanonicalParams, NumType})

    a = vs[ids.a[i]]

    pp_kl_r = KL.gen_normal_kl(pp.r_mean[i], pp.r_var[i])
    (v, (d_r1, d_r2)) = pp_kl_r(vs[ids.r1[i]], vs[ids.r2[i]])

    accum.v -= v * a
    # accum.d[ids.r1[i], s] -= d_r1 .* a
    # accum.d[ids.r2[i], s] -= d_r2 .* a
    # accum.d[ids.a[i], s] -= v
end


@doc """
Subtract the KL divergence from the prior for a
""" ->
function subtract_kl_a!{NumType <: Number}(
  vs::Vector{NumType}, pp::PriorParams,
  accum::SensitiveFloat{CanonicalParams, NumType})
    pp_kl_a = KL.gen_categorical_kl(pp.a)
    (v, (d_a,)) = pp_kl_a(vs[ids.a])
    accum.v -= v
    # accum.d[ids.a, s] -= d_a
end


# @doc """
# Subtract from accum the entropy and expected prior of
# the variational distribution.
# """ ->
# function subtract_kl_value!{NumType <: Number}(
#   mp::ModelParams{NumType}, accum::SensitiveFloat{CanonicalParams, NumType})
#     for s in mp.active_sources
#         subtract_kl_a!(s, mp, accum)
#
#         for i in 1:Ia
#             subtract_kl_r!(i, s, mp, accum)
#             subtract_kl_k!(i, s, mp, accum)
#             for d in 1:D
#                 subtract_kl_c!(d, i, s, mp, accum)
#             end
#         end
#     end
# end


function subtract_kl!{NumType <: Number}(
    mp::ModelParams{NumType}, accum::SensitiveFloat{CanonicalParams, NumType};
    calculate_hessian::Bool=true)

  # The KL divergence as a function of the active source variational parameters.
  function subtract_kl_value_wrapper{NumType <: Number}(vp_vec::Vector{NumType})
    vp_active = reshape(vp_vec, length(CanonicalParams), length(mp.active_sources))
    for sa in 1:length(mp.active_sources)
      vs = vp_active[:, sa]
      subtract_kl_a!(vs, mp.pp, accum)

      for i in 1:Ia
          subtract_kl_r!(i, vs, pp, accum)
          subtract_kl_k!(i, vs, pp, accum)
          for d in 1:D
              subtract_kl_c!(d, i, vs, pp, accum)
          end
      end
    end
  end

  # TODO: do this efficiently with a single function.
  subtract_kl_gradient = ForwardDiff.gradient(subtract_kl_value_wrapper);
  subtract_kl_hessian = ForwardDiff.hessian(subtract_kl_value_wrapper);

  vp_vec = zeros(NumType, length(CanonicalParams), mp.S)
  accum.v = subtract_kl_value_wrapper()

end
