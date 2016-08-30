import ForwardDiff

using Celeste: Model, SensitiveFloats, ElboDeriv
using Distributions
import ElboDeriv: BvnComponent, GalaxyCacheComponent
import ElboDeriv: eval_bvn_pdf!, get_bvn_derivs!, transform_bvn_derivs!


function test_elbo()
    blob, ea, body = gen_two_body_dataset()
    keep_pixels = 10:11
    trim_tiles!(ea.images, keep_pixels)

    # vp_vec is a vector of the parameters from all the active sources.
    function wrap_elbo{NumType <: Number}(vp_vec::Vector{NumType})
        ea_local = unwrap_vp_vector(vp_vec, ea)
        elbo = ElboDeriv.elbo(ea_local, calculate_derivs=false)
        elbo.v[1]
    end

    ea.active_sources = [1]
    vp_vec = wrap_vp_vector(ea, true)
    elbo_1 = ElboDeriv.elbo(ea)
    test_with_autodiff(wrap_elbo, vp_vec, elbo_1)
    #test_elbo_mp(ea, elbo_1)

    ea.active_sources = [2]
    vp_vec = wrap_vp_vector(ea, true)
    elbo_2 = ElboDeriv.elbo(ea)
    test_with_autodiff(wrap_elbo, vp_vec, elbo_2)
    #test_elbo_mp(ea, elbo_2)

    ea.active_sources = [1, 2]
    vp_vec = wrap_vp_vector(ea, true)
    elbo_12 = ElboDeriv.elbo(ea)
    test_with_autodiff(wrap_elbo, vp_vec, elbo_12)

    P = length(CanonicalParams)
    @test size(elbo_1.d) == size(elbo_2.d) == (P, 1)
    @test size(elbo_12.d) == (length(CanonicalParams), 2)

    @test size(elbo_1.h) == size(elbo_2.h) == (P, P)
    @test size(elbo_12.h) == size(elbo_12.h) == (2 * P, 2 * P)
end


function test_real_image()
    # TODO: replace this with stamp tests having non-trivial WCS transforms.
    # TODO: streamline the creation of small real images.

    run, camcol, field = (3900, 6, 269)

    images = SDSSIO.load_field_images(RunCamcolField(run, camcol, field), datadir)
    tiled_images = TiledImage[TiledImage(img) for img in images]
    dir = joinpath(datadir, "$run/$camcol/$field")
    fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir run camcol field
    catalog = SDSSIO.read_photoobj_celeste(fname)

    # Pick an object.
    objid = "1237662226208063499"
    objids = [ce.objid for ce in catalog]
    sa = findfirst(objids, objid)
    neighbors = Infer.find_neighbors([sa], catalog, tiled_images)[1]

    cat_local = vcat(catalog[sa], catalog[neighbors])
    vp = Vector{Float64}[init_source(ce) for ce in cat_local]
    patches, tile_source_map = Infer.get_tile_source_map(tiled_images, cat_local)
    ea = ElboArgs(tiled_images, vp, tile_source_map, patches, [1])
    Infer.fit_object_psfs!(ea, ea.active_sources)
    Infer.trim_source_tiles!(ea)

    elbo = ElboDeriv.elbo(ea)

    function wrap_elbo{NumType <: Number}(vs1::Vector{NumType})
        ea_local = forward_diff_model_params(NumType, ea)
        ea_local.vp[1][:] = vs1
        local_elbo = ElboDeriv.elbo(ea_local, calculate_derivs=false)
        local_elbo.v[1]
    end

    test_with_autodiff(wrap_elbo, ea.vp[1], elbo)
end


function test_transform_sensitive_float()
	blob, ea, body = gen_two_body_dataset();

	# Only keep a few pixels to make the autodiff results faster.
  keep_pixels = 10:11
	for b = 1:ea.N
	  pixels1 = ea.images[b].tiles[1,1].pixels
      h_width, w_width = size(pixels1)
	  pixels1[setdiff(1:h_width, keep_pixels), :] = NaN;
	  pixels1[:, setdiff(1:w_width, keep_pixels)] = NaN;
	end


	function wrap_elbo{NumType <: Number}(vp_free_vec::Vector{NumType})
		vp_free_array = reshape(vp_free_vec, length(UnconstrainedParams), length(ea.active_sources))
		vp_free = Vector{NumType}[ zeros(NumType, length(UnconstrainedParams)) for
		                           sa in ea.active_sources ];
		Transform.array_to_free_vp!(vp_free_array, vp_free, Int[])
		ea_local = forward_diff_model_params(NumType, ea);
		transform.to_vp!(vp_free, ea_local.vp)
		elbo = ElboDeriv.elbo(ea_local, calculate_derivs=false)
		elbo.v[1]
	end

	transform = Transform.get_mp_transform(ea, loc_width=1.0);
	elbo = ElboDeriv.elbo(ea);
	elbo_trans = transform.transform_sensitive_float(elbo, ea);

	free_vp_vec = reduce(vcat, transform.from_vp(ea.vp));
	ad_grad = ForwardDiff.gradient(wrap_elbo, free_vp_vec);
	ad_hess = ForwardDiff.hessian(wrap_elbo, free_vp_vec);

	@test_approx_eq ad_grad reduce(vcat, elbo_trans.d)
	@test_approx_eq ad_hess elbo_trans.h

  # Test with a subset of sources.
	ea.active_sources = [2]
	transform = Transform.get_mp_transform(ea, loc_width=1.0);
	elbo = ElboDeriv.elbo(ea);
	elbo_trans = transform.transform_sensitive_float(elbo, ea);

	free_vp_vec = reduce(vcat, transform.from_vp(ea.vp));
	ad_grad = ForwardDiff.gradient(wrap_elbo, free_vp_vec);
	ad_hess = ForwardDiff.hessian(wrap_elbo, free_vp_vec);

	@test_approx_eq ad_grad reduce(vcat, elbo_trans.d)
	@test_approx_eq ad_hess elbo_trans.h
end


# ForwardDiff 0.2's compilation time is very slow, so only run these tests
# if explicitly requested.

@time test_elbo()
@time test_real_image()
@time test_transform_sensitive_float()
