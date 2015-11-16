using Celeste
using CelesteTypes
using Base.Test
using Distributions
using SampleData
using Transform

import Synthetic
import WCS

println("Running hessian tests.")

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
  @test_approx_eq sf.hs[2, 3, 1] 5.0
  @test_approx_eq sf.hs[3, 2, 1] 5.0

  CelesteTypes.set_hess!(sf, 4, 4, 6.0);
  @test_approx_eq sf.hs[4, 4, 1] 6.0

  CelesteTypes.set_hess!(sf, 2, 3, 2, 7.0);
  @test_approx_eq sf.hs[2, 3, 2] 7.0
  @test_approx_eq sf.hs[3, 2, 2] 7.0

  CelesteTypes.set_hess!(sf, 4, 4, 2, 8.0);
  @test_approx_eq sf.hs[4, 4, 2] 8.0
end


test_set_hess()
