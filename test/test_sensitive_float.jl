using Celeste
using Base.Test
using SampleData
using CelesteTypes
using Compat


function test_combine_sfs()
  # TODO: this test was designed for multiply_sf.  Make it more general.

  # Two sets of ids with some overlap and some disjointness.
  p = length(ids)
  S = 2

  ids1 = find((1:p) .% 2 .== 0)
  ids2 = setdiff(1:p, ids1)
  ids1 = union(ids1, 1:5)
  ids2 = union(ids2, 1:5)

  l1 = zeros(Float64, S * p);
  l2 = zeros(Float64, S * p);
  l1[ids1] = rand(length(ids1))
  l2[ids2] = rand(length(ids2))
  l1[ids1 + p] = rand(length(ids1))
  l2[ids2 + p] = rand(length(ids2))

  sigma1 = zeros(Float64, S * p, S * p);
  sigma2 = zeros(Float64, S * p, S * p);
  sigma1[ids1, ids1] = rand(length(ids1), length(ids1));
  sigma2[ids2, ids2] = rand(length(ids2), length(ids2));
  sigma1[ids1 + p, ids1 + p] = rand(length(ids1), length(ids1));
  sigma2[ids2 + p, ids2 + p] = rand(length(ids2), length(ids2));
  sigma1 = 0.5 * (sigma1 + sigma1');
  sigma2 = 0.5 * (sigma2 + sigma2');

  x = 0.1 * rand(S * p);

  function base_fun1{T <: Number}(x::Vector{T})
    (l1' * x + 0.5 * x' * sigma1 * x)[1,1]
  end

  function base_fun2{T <: Number}(x::Vector{T})
    (l2' * x + 0.5 * x' * sigma2 * x)[1,1]
  end

  function multiply_fun{T <: Number}(x::Vector{T})
    base_fun1(x) * base_fun1(x)
  end

  function combine_fun{T <: Number}(x::Vector{T})
    (base_fun1(x) ^ 2) * sqrt(base_fun2(x))
  end

  function combine_fun_derivatives{T <: Number}(x::Vector{T})
    g_d = T[2 * base_fun1(x) * sqrt(base_fun2(x)),
            0.5 * (base_fun1(x) ^ 2) / sqrt(base_fun2(x)) ]
    g_h = zeros(T, 2, 2)
    g_h[1, 1] = 2 * sqrt(base_fun2(x))
    g_h[2, 2] = -0.25 * (base_fun1(x) ^ 2) * (base_fun2(x) ^(-3/2))
    g_h[1, 2] = g_h[2, 1] = base_fun1(x) / sqrt(base_fun2(x))
    g_d, g_h
  end

  s_ind = Array(UnitRange{Int}, 2);
  s_ind[1] = 1:p
  s_ind[2] = (1:p) + p

  ret1 = zero_sensitive_float(CanonicalParams, Float64, S);
  ret1.v[1] = base_fun1(x)
  fill!(ret1.d, 0.0);
  fill!(ret1.h, 0.0);
  for s=1:S
    ret1.d[:, s] = l1[s_ind[s]] + sigma1[s_ind[s], s_ind[s]] * x[s_ind[s]];
    ret1.h[s_ind[s], s_ind[s]] = sigma1[s_ind[s], s_ind[s]];
  end

  ret2 = zero_sensitive_float(CanonicalParams, Float64, S);
  ret2.v[1] = base_fun2(x)
  fill!(ret2.d, 0.0);
  fill!(ret2.h, 0.0);
  for s=1:S
    ret2.d[:, s] = l2[s_ind[s]] + sigma2[s_ind[s], s_ind[s]] * x[s_ind[s]];
    ret2.h[s_ind[s], s_ind[s]] = sigma2[s_ind[s], s_ind[s]];
  end

  grad = ForwardDiff.gradient(base_fun1, x);
  hess = ForwardDiff.hessian(base_fun1, x);
  for s=1:S
    @test_approx_eq(ret1.d[:, s], grad[s_ind[s]])
  end
  @test_approx_eq(ret1.h, hess)

  grad = ForwardDiff.gradient(base_fun2, x);
  hess = ForwardDiff.hessian(base_fun2, x);
  for s=1:S
    @test_approx_eq(ret2.d[:, s], grad[s_ind[s]])
  end
  @test_approx_eq(ret2.h, hess)

  # Test the combinations.
  v = combine_fun(x);
  grad = ForwardDiff.gradient(combine_fun, x);
  hess = ForwardDiff.hessian(combine_fun, x);

  sf1 = deepcopy(ret1);
  sf2 = deepcopy(ret2);
  g_d, g_h = combine_fun_derivatives(x)
  CelesteTypes.combine_sfs!(sf1, sf2, sf1.v[1] ^ 2 * sqrt(sf2.v[1]), g_d, g_h);

  @test_approx_eq sf1.v[1] v
  @test_approx_eq sf1.d[:] grad
  @test_approx_eq sf1.h hess
end


function test_add_sources_sf()
  P = length(CanonicalParams)
  S = 2

  sf_all = zero_sensitive_float(CanonicalParams, Float64, S);
  sf_s = zero_sensitive_float(CanonicalParams, Float64, 1);

  function scaled_exp!{NumType <: Number}(
      sf::SensitiveFloat{CanonicalParams, NumType},
      x::Vector{NumType}, a::Vector{Float64})

    sf.v[1] = one(NumType)
    for p in 1:P
      sf.v[1] *= exp(sum(x[p] * a[p]))
    end
    if NumType == Float64
      sf.d[:, 1] = sf.v[1] * a
      sf.h[:, :] = sf.v[1] * (a * a')
    end
  end

  a1 = rand(P);
  function f1{NumType <: Number}(x::Vector{NumType})
    sf_local = zero_sensitive_float(CanonicalParams, NumType, 1);
    scaled_exp!(sf_local, x, a1)
    sf_local.v[1]
  end

  a2 = rand(P);
  function f2{NumType <: Number}(x::Vector{NumType})
    sf_local = zero_sensitive_float(CanonicalParams, NumType, 1);
    scaled_exp!(sf_local, x, a2)
    sf_local.v[1]
  end

  x1 = rand(P);
  clear!(sf_s);
  scaled_exp!(sf_s, x1, a1);
  v1 = sf_s.v[1]

  fd_grad1 = ForwardDiff.gradient(f1, x1);
  @test_approx_eq sf_s.d fd_grad1

  fd_hess1 = ForwardDiff.hessian(f1, x1);
  @test_approx_eq sf_s.h fd_hess1

  CelesteTypes.add_sources_sf!(sf_all, sf_s, 1, true)

  x2 = rand(P);
  clear!(sf_s);
  scaled_exp!(sf_s, x2, a2);
  v2 = sf_s.v

  fd_grad2 = ForwardDiff.gradient(f2, x2);
  @test_approx_eq sf_s.d fd_grad2

  fd_hess2 = ForwardDiff.hessian(f2, x2);
  @test_approx_eq sf_s.h fd_hess2

  CelesteTypes.add_sources_sf!(sf_all, sf_s, 2, true)

  @test_approx_eq (v1 + v2) sf_all.v

  @test_approx_eq (v1 + v2) sf_all.v
  @test_approx_eq fd_grad1 sf_all.d[1:P, 1]
  @test_approx_eq fd_grad2 sf_all.d[1:P, 2]
  @test_approx_eq fd_hess1 sf_all.h[1:P, 1:P]
  @test_approx_eq fd_hess2 sf_all.h[(1:P) + P, (1:P) + P]
  @test_approx_eq zeros(P, P) sf_all.h[(1:P), (1:P) + P]
  @test_approx_eq zeros(P, P) sf_all.h[(1:P) + P, (1:P)]
end


test_combine_sfs()
test_add_sources_sf()
