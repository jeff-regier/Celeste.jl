

#trans = Transform.pixel_rect_transform;
trans = Transform.free_transform

blob, mp, body = gen_sample_galaxy_dataset(perturb=true);
OptimizeElbo.maximize_elbo(blob, mp, trans);
verify_sample_galaxy(mp.vp[1], [8.5, 9.6]);

omitted_ids = Int64[];
kept_ids = collect(1:length(ids_free));
obj_wrap = OptimizeElbo.ObjectiveWrapperFunctions(
    mp -> ElboDeriv.elbo(blob, mp), deepcopy(mp), trans, kept_ids, omitted_ids);

x = trans.vp_to_vector(mp.vp, omitted_ids);
celeste_grad = obj_wrap.f_grad(x);
ad_grad = obj_wrap.f_ad_grad(x);
DataFrame(name=ids_free_names[kept_ids], f=celeste_grad, ad=ad_grad, diff=celeste_grad - ad_grad)
ad_hess = obj_wrap.f_ad_hessian(x);

newton_p = -ad_hess \ ad_grad;
grad_p = ad_grad;

v0 = obj_wrap.f_value(x)
alpha = 0.01; obj_wrap.f_value(x + alpha * grad_p) - v0