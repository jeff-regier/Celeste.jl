using Celeste
using CelesteTypes
using Base.Test
using Distributions
using SampleData
using Transform

import Synthetic
import WCS

println("Running hessian tests.")


# Test for hessians.

ret1 = zero_sensitive_float(CanonicalParams, Float64, 2);
ret2 = zero_sensitive_float(CanonicalParams, Float64, 2);

# Two sets of ids with some overlap and some disjointness.
p = length(ids)
ids1 = find((1:p) .% 2 .== 0)
ids2 = setdiff(1:p, ids1)
ids1 = union(ids1, 1:5)
ids2 = union(ids2, 1:5)

l1 = l2 = zeros(Float64, p);
l1[ids1] = rand(length(ids1))
l2[ids2] = rand(length(ids2))

sigma1 = sigma2 = zeros(Float64, p, p);
sigma1[ids1, ids1] = rand(length(ids1), length(ids1))
sigma2[ids2, ids2] = rand(length(ids2), length(ids2))

x = 0.1 * rand(p);

function testfun1(x)
  (l1' * x + 0.5 * x' * sigma1 * x)[1,1]
end

function testfun2(x)
  (l2' * x + 0.5 * x' * sigma2 * x)[1,1]
end

function testfun(x)
  testfun1(x) * testfun2(x)
end

testfun(x)

hess = zeros(Float64, p, p);

using ForwardDiff
ForwardDiff.hessian!(hess, testfun, x)


# We will test the function ret1 * l1 * sigma * l2 * ret2.
ret1.v = 5.0;
ret2.v = 6.0;




#function test_brightness_hessian()
    blob, mp, three_bodies = gen_three_body_dataset();
    kept_ids = [ ids_free.r1; ids_free.r2; ids_free.c1[:]; ids_free.c2[:] ];
    omitted_ids = setdiff(1:length(ids_free), kept_ids);

    # mp = deepcopy(mp_original);
    # mp.vp = fill(mp_original.vp[1], 1);
    # mp.patches = mp_original.patches[1, :];
    # mp.active_sources = [1]
    # mp.objids = fill(mp_original.objids[1], 1)
    # mp.S = 1

    transform = Transform.get_identity_transform(length(ids), mp.S);
    ad_hess = zeros(Float64, length(kept_ids) * mp.S, length(kept_ids) * mp.S);
    for i = 1:Ia
        for b = [3,4,2,5,1]
            # Write the brightness as an ordinary function of a ModelParams
            # so we can use the objective function autodiff hessian logic.
            function wrap_source_brightness{NumType <: Number}(
                mp::ModelParams{NumType})
              ret = zero_sensitive_float(CanonicalParams, NumType, mp.S);
              for s=1:mp.S
                sb = ElboDeriv.SourceBrightness(mp.vp[s])
                ret.v = sb.E_l_a[b, i].v
                ret.d[:, s] = sb.E_l_a[b, i].d
                ret.hs[:, :, s] = sb.E_l_a[b, i].hs[:, :, 1]
              end
              ret
            end

            bright = wrap_source_brightness(mp);
            objective = OptimizeElbo.ObjectiveWrapperFunctions(
              wrap_source_brightness, mp, transform, kept_ids, omitted_ids);
            x = transform.vp_to_array(mp.vp, omitted_ids);

            # Sanity check.
            @test_approx_eq objective.f_value(x[:]) bright.v

            # Compare the AD hessian with the exact hessian.
            println("b $b i $i")
            objective.f_ad_hessian!(x[:], ad_hess);
            for s=1:mp.S
              hess_ind = (1:length(kept_ids)) + (s - 1) * length(kept_ids)
              @test_approx_eq ad_hess[hess_ind, hess_ind] bright.hs[kept_ids, kept_ids, s]
            end

            reduce(hcat, ind2sub(size(bright.hs[kept_ids, kept_ids]), find(abs(ad_hess[hess_ind, hess_ind][:] .- bright.hs[kept_ids,kept_ids,1][:]) .> 1e-6)))
        end
    end
    #
    #
    #
    #         function wrap_source_brightness_3(mp)
    #             sb = ElboDeriv.SourceBrightness(mp.vp[1])
    #             ret = zero_sensitive_float(CanonicalParams, 3)
    #             ret.v = sb.E_ll_a[b, i].v
    #             ret.d[:, 1] = sb.E_ll_a[b, i].d
    #             ret
    #         end
    #         test_by_finite_differences(wrap_source_brightness_3, mp0)
    #     end
    # end
#end




function test_set_hess()
  sf = zero_sensitive_float(CanonicalParams, 2);
  CelesteTypes.set_hess!(sf, 2, 3, 5.0);
  @test_approx_eq sf.hs[1][2, 3] 5.0
  @test_approx_eq sf.hs[1][3, 2] 5.0

  CelesteTypes.set_hess!(sf, 4, 4, 6.0);
  @test_approx_eq sf.hs[1][4, 4] 6.0

  CelesteTypes.set_hess!(sf, 2, 3, 2, 7.0);
  @test_approx_eq sf.hs[2][2, 3] 7.0
  @test_approx_eq sf.hs[2][3, 2] 7.0

  CelesteTypes.set_hess!(sf, 4, 4, 2, 8.0);
  @test_approx_eq sf.hs[2][4, 4] 8.0
end


test_set_hess()
