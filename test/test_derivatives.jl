import DualNumbers
import ForwardDiff

using Celeste: Model, SensitiveFloats, ElboDeriv
import ElboDeriv: BvnComponent, GalaxyCacheComponent
import ElboDeriv: eval_bvn_pdf!, get_bvn_derivs!, transform_bvn_derivs!

include("derivative_utils.jl")


#######################
# Functions that will be tested each time.

function test_set_hess()
    sf = zero_sensitive_float(CanonicalParams)
    set_hess!(sf, 2, 3, 5.0)
    @test_approx_eq sf.h[2, 3] 5.0
    @test_approx_eq sf.h[3, 2] 5.0

    set_hess!(sf, 4, 4, 6.0)
    @test_approx_eq sf.h[4, 4] 6.0
end


function test_bvn_cov()
        e_axis = .7
        e_angle = pi/5
        e_scale = 2.

        manual_11 = e_scale^2 * (1 + (e_axis^2 - 1) * (sin(e_angle))^2)
        util_11 = ElboDeriv.get_bvn_cov(e_axis, e_angle, e_scale)[1,1]
        @test_approx_eq util_11 manual_11

        manual_12 = e_scale^2 * (1 - e_axis^2) * (cos(e_angle)sin(e_angle))
        util_12 = ElboDeriv.get_bvn_cov(e_axis, e_angle, e_scale)[1,2]
        @test_approx_eq util_12 manual_12

        manual_22 = e_scale^2 * (1 + (e_axis^2 - 1) * (cos(e_angle))^2)
        util_22 = ElboDeriv.get_bvn_cov(e_axis, e_angle, e_scale)[2,2]
        @test_approx_eq util_22 manual_22
end


function test_real_image()
    # TODO: replace this with stamp tests having non-trivial WCS transforms.
    # TODO: streamline the creation of small real images.

    run, camcol, field = (3900, 6, 269)

    images = SDSSIO.load_field_images(run, camcol, field, datadir)
    tiled_images = TiledImage[TiledImage(img) for img in images]
    fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" datadir run camcol field
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

    ad_grad = ForwardDiff.gradient(wrap_elbo, ea.vp[1]);
    ad_hess = ForwardDiff.hessian(wrap_elbo, ea.vp[1]);

    # Sanity check
    ad_v = wrap_elbo(ea.vp[1]);
    @test_approx_eq ad_v elbo.v

    hcat(ad_grad, elbo.d[:, 1])
    @test_approx_eq ad_grad elbo.d[:, 1]
    @test_approx_eq ad_hess elbo.h
end


function test_dual_numbers()
    # Simply check that the likelihood can be used with dual numbers.
    # Due to the autodiff parts of the KL divergence and transform,
    # these parts of the ELBO will currently not work with dual numbers.
    blob, ea, body = gen_sample_star_dataset()
    ea_dual = forward_diff_model_params(DualNumbers.Dual{Float64}, ea)
    elbo_dual = ElboDeriv.elbo_likelihood(ea_dual)

    true
end


function test_tile_predicted_image()
    blob, ea, body = gen_sample_star_dataset(perturb=false)
    tile = ea.images[1].tiles[1, 1]
    tile_source_map = ea.tile_source_map[1][1, 1]
    pred_image =
        ElboDeriv.tile_predicted_image(tile, ea, tile_source_map; include_epsilon=true)

    # Regress the tile pixels onto the predicted image
    # TODO: Why isn't the regression closer to one?    Something in the sample data
    # generation?
    reg_coeff = dot(tile.pixels[:], pred_image[:]) / dot(pred_image[:], pred_image[:])
    residuals = pred_image * reg_coeff - tile.pixels
    residual_sd = sqrt(mean(residuals .^ 2))

    @test residual_sd / mean(tile.pixels) < 0.1
end


function test_derivative_flags()
    blob, ea, body = gen_two_body_dataset()
    keep_pixels = 10:11
    trim_tiles!(ea.images, keep_pixels)

    elbo = ElboDeriv.elbo(ea)

    elbo_noderiv = ElboDeriv.elbo(ea; calculate_derivs=false)
    @test_approx_eq elbo.v[1] elbo_noderiv.v
    @test_approx_eq elbo_noderiv.d zeros(size(elbo_noderiv.d))
    @test_approx_eq elbo_noderiv.h zeros(size(elbo_noderiv.h))

    elbo_nohess = ElboDeriv.elbo(ea; calculate_hessian=false)
    @test_approx_eq elbo.v[1] elbo_nohess.v
    @test_approx_eq elbo.d elbo_nohess.d
    @test_approx_eq elbo_noderiv.h zeros(size(elbo_noderiv.h))
end


function test_active_sources()
    # Test that the derivatives of the expected brightnesses partition in
    # active_sources.

    blob, ea, body = gen_two_body_dataset()
    keep_pixels = 10:11
    trim_tiles!(ea.images, keep_pixels)
    b = 1
    tile = ea.images[b].tiles[1,1]
    h, w = 10, 10

    ea.active_sources = [1, 2]
    elbo_lik_12 = ElboDeriv.elbo_likelihood(ea)

    ea.active_sources = [1]
    elbo_lik_1 = ElboDeriv.elbo_likelihood(ea)

    ea.active_sources = [2]
    elbo_lik_2 = ElboDeriv.elbo_likelihood(ea)

    @test_approx_eq elbo_lik_12.v[1] elbo_lik_1.v
    @test_approx_eq elbo_lik_12.v[1] elbo_lik_2.v

    @test_approx_eq elbo_lik_12.d[:, 1] elbo_lik_1.d[:, 1]
    @test_approx_eq elbo_lik_12.d[:, 2] elbo_lik_2.d[:, 1]

    P = length(CanonicalParams)
    @test_approx_eq elbo_lik_12.h[1:P, 1:P] elbo_lik_1.h
    @test_approx_eq elbo_lik_12.h[(1:P) + P, (1:P) + P] elbo_lik_2.h
end


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
    #test_elbo_mp(ea, elbo_12)

    P = length(CanonicalParams)
    @test size(elbo_1.d) == size(elbo_2.d) == (P, 1)
    @test size(elbo_12.d) == (length(CanonicalParams), 2)

    @test size(elbo_1.h) == size(elbo_2.h) == (P, P)
    @test size(elbo_12.h) == size(elbo_12.h) == (2 * P, 2 * P)
end


############################################
# The tests below are currently very slow to compile due to changes
# in ForwardDiff in v0.2.2, so they will only be enabled optionally..

"""
This is the function of which get_bvn_derivs!() returns the derivatives.
It is only used for testing.
"""
function eval_bvn_log_density{NumType <: Number}(
        elbo_vars::ElboDeriv.ElboIntermediateVariables{NumType},
        bvn::BvnComponent{NumType}, x::Vector{Float64})

    ElboDeriv.eval_bvn_pdf!(elbo_vars.bvn_derivs, bvn, x)

    -0.5 * (
        (x[1] - bvn.the_mean[1]) * elbo_vars.bvn_derivs.py1[1] +
        (x[2] - bvn.the_mean[2]) * elbo_vars.bvn_derivs.py2[1] -
        log(bvn.precision[1, 1] * bvn.precision[2, 2] - bvn.precision[1, 2] ^ 2))
end


function test_process_active_pixels()
    blob, ea, bodies = gen_two_body_dataset()

    # Choose four pixels only to keep the test fast.
    active_pixels = Array(ElboDeriv.ActivePixel, 4)
    active_pixels[1] = ActivePixel(1, 1, 10, 11)
    active_pixels[2] = ActivePixel(1, 1, 11, 10)
    active_pixels[3] = ActivePixel(5, 1, 10, 11)
    active_pixels[4] = ActivePixel(5, 1, 11, 10)


    function tile_lik_wrapper_fun{NumType <: Number}(
            ea::ElboArgs{NumType}, calculate_derivs::Bool)

        elbo_vars = ElboIntermediateVariables(NumType, ea.S,
                                            length(ea.active_sources),
                                            calculate_derivs=calculate_derivs,
                                            calculate_hessian=calculate_hessian)
        ElboDeriv.process_active_pixels!(elbo_vars, ea.images, ea, active_pixels)
        deepcopy(elbo_vars.elbo)
    end

    function tile_lik_value_wrapper{NumType <: Number}(x::Vector{NumType})
        ea_local = unwrap_vp_vector(x, ea)
        tile_lik_wrapper_fun(ea_local, false).v[1]
    end

    elbo = tile_lik_wrapper_fun(ea, true)

    x = wrap_vp_vector(ea, true)
    test_with_autodiff(tile_lik_value_wrapper, x, elbo)
end


function test_add_log_term()
    blob, ea, bodies = gen_two_body_dataset()

    # Test this pixel
    h, w = (10, 10)

    for b = 1:5
        println("Testing log term for band $b.")
        x_nbm = 70.
        tile = ea.images[b].tiles[1,1]
        tile_source_map = ea.tile_source_map[b][1,1]

        iota = median(blob[b].iota_vec)

        function add_log_term_wrapper_fun{NumType <: Number}(
                ea::ElboArgs{NumType}, calculate_derivs::Bool)

            star_mcs, gal_mcs =
                ElboDeriv.load_bvn_mixtures(ea, b, calculate_derivs=calculate_derivs)
            sbs = ElboDeriv.SourceBrightness{NumType}[
                ElboDeriv.SourceBrightness(ea.vp[s], calculate_derivs=calculate_derivs)
                for s in 1:ea.S]

            elbo_vars_loc = ElboDeriv.ElboIntermediateVariables(NumType, ea.S, ea.S)
            elbo_vars_loc.calculate_derivs = calculate_derivs
            ElboDeriv.populate_fsm_vecs!(
                elbo_vars_loc, ea, tile_source_map, tile, h, w, sbs, gal_mcs, star_mcs)
            ElboDeriv.combine_pixel_sources!(
                elbo_vars_loc, ea, tile_source_map, tile, sbs)

            ElboDeriv.add_elbo_log_term!(elbo_vars_loc, x_nbm, iota)

            deepcopy(elbo_vars_loc.elbo)
        end

        function ad_wrapper_fun{NumType <: Number}(x::Vector{NumType})
            ea_local = unwrap_vp_vector(x, ea)
            add_log_term_wrapper_fun(ea_local, false).v[1]
        end

        x = wrap_vp_vector(ea, true)
        elbo = add_log_term_wrapper_fun(ea, true)
        test_with_autodiff(ad_wrapper_fun, x, elbo)
    end
end


function test_combine_pixel_sources()
    blob, ea, bodies = gen_two_body_dataset()

    S = length(ea.active_sources)
    P = length(CanonicalParams)
    h = 10
    w = 10

    for test_var = [false, true], b=1:5
        test_var_string = test_var ? "E_G" : "var_G"
        println("Testing $(test_var_string), band $b")

        tile = ea.images[b].tiles[1,1]; # Note: only one tile in this simulated dataset.
        tile_source_map = ea.tile_source_map[b][1,1]

        function e_g_wrapper_fun{NumType <: Number}(
                ea::ElboArgs{NumType}; calculate_derivs=true)

            star_mcs, gal_mcs =
                ElboDeriv.load_bvn_mixtures(ea, b, calculate_derivs=calculate_derivs)
            sbs = ElboDeriv.SourceBrightness{NumType}[
                ElboDeriv.SourceBrightness(ea.vp[s], calculate_derivs=calculate_derivs)
                for s in 1:ea.S]

            elbo_vars_loc = ElboDeriv.ElboIntermediateVariables(NumType, ea.S, ea.S)
            elbo_vars_loc.calculate_derivs = calculate_derivs
            ElboDeriv.populate_fsm_vecs!(
                elbo_vars_loc, ea, tile_source_map, tile, h, w, sbs, gal_mcs, star_mcs)
            ElboDeriv.combine_pixel_sources!(
                elbo_vars_loc, ea, tile_source_map, tile, sbs)
            deepcopy(elbo_vars_loc)
        end

        function wrapper_fun{NumType <: Number}(x::Vector{NumType})
            ea_local = unwrap_vp_vector(x, ea)
            elbo_vars_local = e_g_wrapper_fun(ea_local, calculate_derivs=false)
            test_var ? elbo_vars_local.var_G.v[1] : elbo_vars_local.E_G.v[1]
        end

        x = wrap_vp_vector(ea, true)
        elbo_vars = e_g_wrapper_fun(ea)
        sf = test_var ? deepcopy(elbo_vars.var_G) : deepcopy(elbo_vars.E_G)

        test_with_autodiff(wrapper_fun, x, sf)
    end
end


function test_e_g_s_functions()
    blob, ea, bodies = gen_two_body_dataset()

    # S = length(ea.active_sources)
    P = length(CanonicalParams)
    h = 10
    w = 10
    s = 1

    for test_var = [false, true], b=1:5
        test_var_string = test_var ? "E_G" : "var_G"
        println("Testing $(test_var_string), band $b")

        tile = ea.images[b].tiles[1,1]; # Note: only one tile in this simulated dataset.
        tile_source_map = ea.tile_source_map[b][1,1]

        function e_g_wrapper_fun{NumType <: Number}(
                ea::ElboArgs{NumType}; calculate_derivs=true)

            star_mcs, gal_mcs =
                ElboDeriv.load_bvn_mixtures(ea, b, calculate_derivs=calculate_derivs)
            sbs = ElboDeriv.SourceBrightness{NumType}[
                ElboDeriv.SourceBrightness(ea.vp[s], calculate_derivs=calculate_derivs)
                for s in 1:ea.S]

            elbo_vars_loc = ElboDeriv.ElboIntermediateVariables(
                NumType, ea.S, length(ea.active_sources))
            elbo_vars_loc.calculate_derivs = calculate_derivs
            ElboDeriv.populate_fsm_vecs!(
                elbo_vars_loc, ea, tile_source_map, tile, h, w, sbs, gal_mcs, star_mcs)
            ElboDeriv.accumulate_source_brightness!(elbo_vars_loc, ea, sbs, s, b)
            deepcopy(elbo_vars_loc)
        end

        function wrapper_fun{NumType <: Number}(x::Vector{NumType})
            @assert length(x) == P
            ea_local = forward_diff_model_params(NumType, ea)
            ea_local.vp[s] = x
            elbo_vars_local = e_g_wrapper_fun(ea_local, calculate_derivs=false)
            test_var ? elbo_vars_local.var_G_s.v[1] : elbo_vars_local.E_G_s.v[1]
        end

        x = ea.vp[s]

        elbo_vars = e_g_wrapper_fun(ea)

        # Sanity check the variance value.
        @test_approx_eq(elbo_vars.var_G_s.v,
                                        elbo_vars.E_G2_s.v[1] - (elbo_vars.E_G_s.v[1] ^ 2))

        sf = test_var ? deepcopy(elbo_vars.var_G_s) : deepcopy(elbo_vars.E_G_s)

        test_with_autodiff(wrapper_fun, x, sf)
    end
end


function test_fs1m_derivatives()
    # TODO: test with a real and asymmetric wcs jacobian.
    blob, ea, three_bodies = gen_three_body_dataset()
    omitted_ids = Int[]
    kept_ids = setdiff(1:length(ids), omitted_ids)

    s = 1
    b = 1

    patch = ea.patches[s, b]
    u = ea.vp[s][ids.u]
    u_pix = WCSUtils.world_to_pix(
        patch.wcs_jacobian, patch.center, patch.pixel_center, u)
    x = ceil(u_pix + [1.0, 2.0])

    elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, 1, 1)

    ###########################
    # Galaxies

    # Pick out a single galaxy component for testing.
    # The index is (psf, galaxy, gal type, source)
    for psf_k=1:psf_K, type_i = 1:2, gal_j in 1:[8,6][type_i]
        gcc_ind = (psf_k, gal_j, type_i, s)
        function f_wrap_gal{T <: Number}(par::Vector{T})
            # This uses ea, x, wcs_jacobian, and gcc_ind from the enclosing namespace.
            ea_fd = forward_diff_model_params(T, ea)
            elbo_vars_fd = ElboDeriv.ElboIntermediateVariables(T, 1, 1)

            # Make sure par is as long as the galaxy parameters.
            @assert length(par) == length(shape_standard_alignment[2])
            for p1 in 1:length(par)
                    p0 = shape_standard_alignment[2][p1]
                    ea_fd.vp[s][p0] = par[p1]
            end
            star_mcs, gal_mcs =
                ElboDeriv.load_bvn_mixtures(ea_fd, b, calculate_derivs=false)

            # Raw:
            gcc = gal_mcs[gcc_ind...]
            eval_bvn_pdf!(elbo_vars_fd.bvn_derivs, gcc.bmc, x)
            elbo_vars_fd.bvn_derivs.f_pre[1] * gcc.e_dev_i
        end

        function ea_to_par_gal(ea::ElboArgs{Float64})
            par = zeros(length(shape_standard_alignment[2]))
            for p1 in 1:length(par)
                    p0 = shape_standard_alignment[2][p1]
                    par[p1] = ea.vp[s][p0]
            end
            par
        end

        par_gal = ea_to_par_gal(ea)

        star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(ea, b)
        clear!(elbo_vars.fs1m_vec[s])
        ElboDeriv.accum_galaxy_pos!(
            elbo_vars, s, gal_mcs[gcc_ind...], x, patch.wcs_jacobian, true)
        fs1m = deepcopy(elbo_vars.fs1m_vec[s])

        # Two sanity checks.
        gcc = gal_mcs[gcc_ind...]
        clear!(elbo_vars.fs1m_vec[s])
        v = eval_bvn_log_density(elbo_vars, gcc.bmc, x)
        gc = galaxy_prototypes[gcc_ind[3]][gcc_ind[2]]
        pc = ea.patches[s, b].psf[gcc_ind[1]]

        @test_approx_eq(
            pc.alphaBar * gc.etaBar * gcc.e_dev_i * exp(v) / (2 * pi),
            fs1m.v)

        test_with_autodiff(f_wrap_gal, par_gal, fs1m)
    end
end


function test_fs0m_derivatives()
    # TODO: test with a real and asymmetric wcs jacobian.
    blob, ea, three_bodies = gen_three_body_dataset()
    omitted_ids = Int[]
    kept_ids = setdiff(1:length(ids), omitted_ids)

    s = 1
    b = 1

    patch = ea.patches[s, b]
    u = ea.vp[s][ids.u]
    u_pix = WCSUtils.world_to_pix(
        patch.wcs_jacobian, patch.center, patch.pixel_center, u)
    x = ceil(u_pix + [1.0, 2.0])

    elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, 1, 1)

    ###########################
    # Stars

    # Pick out a single star component for testing.
    # The index is psf, source
    for psf_k=1:psf_K
        bmc_ind = (psf_k, s)
        function f_wrap_star{T <: Number}(par::Vector{T})
            # This uses ea, x, wcs_jacobian, and gcc_ind from the enclosing namespace.
            ea_fd = forward_diff_model_params(T, ea)

            # Make sure par is as long as the galaxy parameters.
            @assert length(par) == length(ids.u)
            for p1 in 1:2
                    p0 = ids.u[p1]
                    ea_fd.vp[s][p0] = par[p1]
            end
            star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(ea_fd, b)
            elbo_vars_fd = ElboDeriv.ElboIntermediateVariables(T, 1, 1)
            ElboDeriv.accum_star_pos!(
                elbo_vars_fd, s, star_mcs[bmc_ind...], x, patch.wcs_jacobian, true)
            elbo_vars_fd.fs0m_vec[s].v[1]
        end

        function ea_to_par_star(ea::ElboArgs{Float64})
            par = zeros(2)
            for p1 in 1:length(par)
                    par[p1] = ea.vp[s][ids.u[p1]]
            end
            par
        end

        par_star = ea_to_par_star(ea)

        clear!(elbo_vars.fs0m_vec[s])
        star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(ea, b)
        ElboDeriv.accum_star_pos!(
            elbo_vars, s, star_mcs[bmc_ind...], x, patch.wcs_jacobian, true)
        fs0m = deepcopy(elbo_vars.fs0m_vec[s])

        test_with_autodiff(f_wrap_star, par_star, fs0m)
    end
end


function test_bvn_derivatives()
    # Test log(bvn prob) / d(mean, sigma)

    x = Float64[2.0, 3.0]

    e_angle, e_axis, e_scale = (1.1, 0.02, 4.8)
    sigma = ElboDeriv.get_bvn_cov(e_axis, e_angle, e_scale)

    offset = Float64[0.5, 0.25]

    # Note that get_bvn_derivs doesn't use the weight, so set it to something
    # strange to check that it doesn't matter.
    weight = 0.724

    bvn = BvnComponent{Float64}(offset, sigma, weight)
    elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, 1, 1)
    ElboDeriv.eval_bvn_pdf!(elbo_vars.bvn_derivs, bvn, x)
    ElboDeriv.get_bvn_derivs!(elbo_vars.bvn_derivs, bvn, true, true)

    function bvn_function{T <: Number}(x::Vector{T}, sigma::Matrix{T})
        local_x = offset - x
        -0.5 * ((local_x' * (sigma \ local_x))[1,1] + log(det(sigma)))
    end

    x_ids = 1:2
    sig_ids = 3:5
    function wrap(x::Vector{Float64}, sigma::Matrix{Float64})
        par = zeros(Float64, length(x_ids) + length(sig_ids))
        par[x_ids] = x
        par[sig_ids] = [ sigma[1, 1], sigma[1, 2], sigma[2, 2]]
        par
    end

    function f_wrap{T <: Number}(par::Vector{T})
        x_loc = par[x_ids]
        s_vec = par[sig_ids]
        sig_loc = T[s_vec[1] s_vec[2]; s_vec[2] s_vec[3]]
        bvn_function(x_loc, sig_loc)
    end

    par = wrap(x, sigma)

    # Sanity check
    @test_approx_eq eval_bvn_log_density(elbo_vars, bvn, x) f_wrap(par)

    bvn_derivs = elbo_vars.bvn_derivs
    ad_grad = ForwardDiff.gradient(f_wrap, par)
    @test_approx_eq bvn_derivs.bvn_x_d ad_grad[x_ids]
    @test_approx_eq bvn_derivs.bvn_sig_d ad_grad[sig_ids]

    ad_hess = ForwardDiff.hessian(f_wrap, par)
    @test_approx_eq bvn_derivs.bvn_xx_h ad_hess[x_ids, x_ids]
    @test_approx_eq bvn_derivs.bvn_xsig_h ad_hess[x_ids, sig_ids]
    @test_approx_eq bvn_derivs.bvn_sigsig_h ad_hess[sig_ids, sig_ids]
end


function test_galaxy_variable_transform()
    # This is testing transform_bvn_derivs!

    # TODO: test with a real and asymmetric wcs jacobian.
    # We only need this for a psf and jacobian.
    blob, ea, three_bodies = gen_three_body_dataset()

    # Pick a single source and band for testing.
    s = 1
    b = 5

    # The pixel and world centers shouldn't matter for derivatives.
    patch = ea.patches[s, b]
    psf = patch.psf[1]

    # Pick out a single galaxy component for testing.
    gp = galaxy_prototypes[2][4]
    e_dev_dir = -1.0
    e_dev_i = 0.85

    # Test the variable transformation.
    e_angle, e_axis, e_scale = (1.1, 0.02, 4.8)

    u = Float64[5.3, 2.9]
    x = Float64[7.0, 5.0]

    # The indices in par of each variable.
    par_ids_u = [1, 2]
    par_ids_e_axis = 3
    par_ids_e_angle = 4
    par_ids_e_scale = 5
    par_ids_length = 5

    function wrap_par{T <: Number}(
            u::Vector{T}, e_angle::T, e_axis::T, e_scale::T)
        par = zeros(T, par_ids_length)
        par[par_ids_u] = u
        par[par_ids_e_angle] = e_angle
        par[par_ids_e_axis] = e_axis
        par[par_ids_e_scale] = e_scale
        par
    end


    function f_bvn_wrap{T <: Number}(par::Vector{T})
        u = par[par_ids_u]
        e_angle = par[par_ids_e_angle]
        e_axis = par[par_ids_e_axis]
        e_scale = par[par_ids_e_scale]
        u_pix = WCSUtils.world_to_pix(
            patch.wcs_jacobian, patch.center, patch.pixel_center, u)

        sigma = ElboDeriv.get_bvn_cov(e_axis, e_angle, e_scale)

        function bvn_function{T <: Number}(u_pix::Vector{T}, sigma::Matrix{T})
            local_x = x - u_pix
            -0.5 * ((local_x' * (sigma \ local_x))[1,1] + log(det(sigma)))
        end

        bvn_function(u_pix, sigma)
    end

    # First just test the bvn function itself
    par = wrap_par(u, e_angle, e_axis, e_scale)
    u_pix = WCSUtils.world_to_pix(
        patch.wcs_jacobian, patch.center, patch.pixel_center, u)
    sigma = ElboDeriv.get_bvn_cov(e_axis, e_angle, e_scale)
    bmc = BvnComponent{Float64}(u_pix, sigma, 1.0)
    sig_sf = ElboDeriv.GalaxySigmaDerivs(e_angle, e_axis, e_scale, sigma)
    gcc = GalaxyCacheComponent(1.0, 1.0, bmc, sig_sf)
    elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, 1, 1)
    ElboDeriv.eval_bvn_pdf!(elbo_vars.bvn_derivs, bmc, x)
    ElboDeriv.get_bvn_derivs!(elbo_vars.bvn_derivs, bmc, true, true)
    ElboDeriv.transform_bvn_derivs!(
        elbo_vars.bvn_derivs, gcc.sig_sf, patch.wcs_jacobian, true)

    f_bvn_wrap(par)

    # Check the gradient.
    ad_grad = ForwardDiff.gradient(f_bvn_wrap, par)
    bvn_derivs = elbo_vars.bvn_derivs
    @test_approx_eq ad_grad [bvn_derivs.bvn_u_d; bvn_derivs.bvn_s_d]

    ad_hess = ForwardDiff.hessian(f_bvn_wrap, par)
    @test_approx_eq ad_hess[1:2, 1:2] bvn_derivs.bvn_uu_h
    @test_approx_eq ad_hess[1:2, 3:5] bvn_derivs.bvn_us_h

    celeste_bvn_ss_h = deepcopy(bvn_derivs.bvn_ss_h)
    ad_bvn_ss_h = deepcopy(ad_hess[3:5, 3:5])
    @test_approx_eq ad_hess[3:5, 3:5] bvn_derivs.bvn_ss_h
end


function test_galaxy_cache_component()
    # TODO: eliminate some of the redundancy in these tests.

    # TODO: test with a real and asymmetric wcs jacobian.
    # We only need this for a psf and jacobian.
    blob, ea, three_bodies = gen_three_body_dataset()

    # Pick a single source and band for testing.
    s = 1
    b = 5

    # The pixel and world centers shouldn't matter for derivatives.
    patch = ea.patches[s, b]
    psf = patch.psf[1]

    # Pick out a single galaxy component for testing.
    gp = galaxy_prototypes[2][4]
    e_dev_dir = -1.0
    e_dev_i = 0.85

    # Test the variable transformation.
    e_angle, e_axis, e_scale = (1.1, 0.02, 4.8)

    u = Float64[5.3, 2.9]
    x = Float64[7.0, 5.0]

    # The indices in par of each variable.
    par_ids_u = [1, 2]
    par_ids_e_axis = 3
    par_ids_e_angle = 4
    par_ids_e_scale = 5
    par_ids_length = 5

    function f_wrap{T <: Number}(par::Vector{T})
        u = par[par_ids_u]
        e_angle = par[par_ids_e_angle]
        e_axis = par[par_ids_e_axis]
        e_scale = par[par_ids_e_scale]
        u_pix = WCSUtils.world_to_pix(
            patch.wcs_jacobian, patch.center, patch.pixel_center, u)
        elbo_vars_fd = ElboDeriv.ElboIntermediateVariables(T, 1, 1)
        e_dev_i_fd = convert(T, e_dev_i)
        gcc = GalaxyCacheComponent(
                        e_dev_dir, e_dev_i_fd, gp, psf,
                        u_pix, e_axis, e_angle, e_scale, false, false)

        eval_bvn_pdf!(elbo_vars_fd.bvn_derivs, gcc.bmc, x)

        log(elbo_vars_fd.bvn_derivs.f_pre[1])
    end

    function wrap_par{T <: Number}(
            u::Vector{T}, e_angle::T, e_axis::T, e_scale::T)
        par = zeros(T, par_ids_length)
        par[par_ids_u] = u
        par[par_ids_e_angle] = e_angle
        par[par_ids_e_axis] = e_axis
        par[par_ids_e_scale] = e_scale
        par
    end

    par = wrap_par(u, e_angle, e_axis, e_scale)
    u_pix = WCSUtils.world_to_pix(
        patch.wcs_jacobian, patch.center, patch.pixel_center, u)
    gcc = GalaxyCacheComponent(
                    e_dev_dir, e_dev_i, gp, psf,
                    u_pix, e_axis, e_angle, e_scale, true, true)
    elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, 1, 1)
    eval_bvn_pdf!(elbo_vars.bvn_derivs, gcc.bmc, x)
    get_bvn_derivs!(elbo_vars.bvn_derivs, gcc.bmc, true, true)
    transform_bvn_derivs!(elbo_vars.bvn_derivs, gcc.sig_sf, patch.wcs_jacobian, true)

    # Sanity check the wrapper.
    @test_approx_eq(
        -0.5 *((x - gcc.bmc.the_mean)' * gcc.bmc.precision * (x - gcc.bmc.the_mean) -
                     log(det(gcc.bmc.precision)))[1,1] - log(2pi) +
                     log(psf.alphaBar * gp.etaBar),
        f_wrap(par))

    # Check the gradient.
    ad_grad_fun = x -> ForwardDiff.gradient(f_wrap, x)
    ad_grad = ad_grad_fun(par)
    bvn_derivs = elbo_vars.bvn_derivs
    @test_approx_eq ad_grad [bvn_derivs.bvn_u_d; bvn_derivs.bvn_s_d]

    ad_hess_fun = x -> ForwardDiff.hessian(f_wrap, x)
    ad_hess = ad_hess_fun(par)

    @test_approx_eq ad_hess[1:2, 1:2] bvn_derivs.bvn_uu_h
    @test_approx_eq ad_hess[1:2, 3:5] bvn_derivs.bvn_us_h

    # I'm not sure why this requires less precision for this test.
    celeste_bvn_ss_h = deepcopy(bvn_derivs.bvn_ss_h)
    ad_bvn_ss_h = deepcopy(ad_hess[3:5, 3:5])
    @test_approx_eq ad_hess[3:5, 3:5] bvn_derivs.bvn_ss_h

end


function test_galaxy_sigma_derivs()
    # Test d sigma / d shape

    e_angle, e_axis, e_scale = (pi / 4, 0.7, 1.2)

    function wrap_par{T <: Number}(e_angle::T, e_axis::T, e_scale::T)
        par = zeros(T, length(gal_shape_ids))
        par[gal_shape_ids.e_angle] = e_angle
        par[gal_shape_ids.e_axis] = e_axis
        par[gal_shape_ids.e_scale] = e_scale
        par
    end

    for si in 1:3
        sig_i = [(1, 1), (1, 2), (2, 2)][si]
        println("Testing sigma[$(sig_i)]")
        function f_wrap{T <: Number}(par::Vector{T})
            e_angle_fd = par[gal_shape_ids.e_angle]
            e_axis_fd = par[gal_shape_ids.e_axis]
            e_scale_fd = par[gal_shape_ids.e_scale]
            this_cov = ElboDeriv.get_bvn_cov(e_axis_fd, e_angle_fd, e_scale_fd)
            this_cov[sig_i...]
        end

        par = wrap_par(e_angle, e_axis, e_scale)
        XiXi = ElboDeriv.get_bvn_cov(e_axis, e_angle, e_scale)

        gal_derivs = ElboDeriv.GalaxySigmaDerivs(e_angle, e_axis, e_scale, XiXi)

        ad_grad_fun = x -> ForwardDiff.gradient(f_wrap, x)
        ad_grad = ad_grad_fun(par)
        @test_approx_eq gal_derivs.j[si, :][:] ad_grad

        ad_hess_fun = x -> ForwardDiff.hessian(f_wrap, x)
        ad_hess = ad_hess_fun(par)
        @test_approx_eq(
            ad_hess,
            reshape(gal_derivs.t[si, :, :],
                            length(gal_shape_ids), length(gal_shape_ids)))
    end
end


function test_brightness_hessian()
    blob, ea, star_cat = gen_sample_star_dataset()
    kept_ids = [ ids.r1; ids.r2; ids.c1[:]; ids.c2[:] ]
    omitted_ids = setdiff(1:length(ids), kept_ids)
    i = 1

    for squares in [false, true], b in 1:5, i in 1:2
        squares_string = squares ? "E_G" : "E_G2"
        println("Testing brightness $(squares_string) for band $b, type $i")
        function wrap_source_brightness{NumType <: Number}(
                vp::Vector{NumType}, calculate_derivs::Bool)

            sb = ElboDeriv.SourceBrightness(vp, calculate_derivs=calculate_derivs)
            if squares
                return deepcopy(sb.E_ll_a[b, i])
            else
                return deepcopy(sb.E_l_a[b, i])
            end
        end

        function wrap_source_brightness_value{NumType <: Number}(
                bright_vp::Vector{NumType})
            vp = zeros(NumType, length(CanonicalParams))
            for b_i in 1:length(brightness_standard_alignment[i])
                vp[brightness_standard_alignment[i][b_i]] = bright_vp[b_i]
            end
            wrap_source_brightness(vp, false).v[1]
        end

        bright_vp = ea.vp[1][brightness_standard_alignment[i]]
        bright = wrap_source_brightness(ea.vp[1], true)

        @test_approx_eq bright.v[1] wrap_source_brightness_value(bright_vp)

        ad_grad = ForwardDiff.gradient(wrap_source_brightness_value, bright_vp)
        @test_approx_eq ad_grad bright.d[:, 1]

        ad_hess = ForwardDiff.hessian(wrap_source_brightness_value, bright_vp)
        @test_approx_eq ad_hess bright.h
    end
end


function test_dsiginv_dsig()
    e_angle, e_axis, e_scale = (1.1, 0.02, 4.8) # bvn_derivs.bvn_sigsig_h is large
    the_cov = ElboDeriv.get_bvn_cov(e_axis, e_angle, e_scale)
    the_mean = Float64[0., 0.]
    bvn = BvnComponent{Float64}(the_mean, the_cov, 1.0)
    sigma_vec = Float64[ the_cov[1, 1], the_cov[1, 2], the_cov[2, 2] ]

    for component_index = 1:3
        components = [(1, 1), (1, 2), (2, 2)]
        function invert_sigma{NumType <: Number}(sigma_vec::Vector{NumType})
            sigma_loc = NumType[sigma_vec[1] sigma_vec[2]; sigma_vec[2] sigma_vec[3]]
            sigma_inv = inv(sigma_loc)
            sigma_inv[components[component_index]...]
        end

        ad_grad = ForwardDiff.gradient(invert_sigma, sigma_vec)
        @test_approx_eq ad_grad bvn.dsiginv_dsig[component_index, :][:]
    end
end


########################
# Test derivatives of KL divergence functions

"""
Use Monte Carlo to check whether KL(q_dist || p_dist) matches exact_kl

Args:
    q_dist, p_dist: Distribution objects
    exact_kl: The expected exact KL
"""
function test_kl_value(q_dist, p_dist, exact_kl::Float64)
    sample_size = 2_000_000

    q_samples = rand(q_dist, sample_size)
    empirical_kl_samples =
      logpdf(q_dist, q_samples) - logpdf(p_dist, q_samples)
    empirical_kl = mean(empirical_kl_samples)
    tol = 4 * std(empirical_kl_samples) / sqrt(sample_size)
    min_diff = 1e-2 * std(empirical_kl_samples) / sqrt(sample_size)

    @test_approx_eq_eps empirical_kl exact_kl tol
end

function test_beta_kl_derivatives()
    alpha2 = 3.5
    beta2 = 4.3

    alpha1 = 4.1
    beta1 = 3.9

    par = Float64[ alpha1, beta1 ]

    beta_kl = gen_beta_kl(alpha2, beta)
    function beta_kl_wrapper{NumType <: Number}(par::Vector{NumType})
        alpha1 = par[1]
        beta1 = par[2]
        return beta_kl(alpha1, beta1, false)
    end

    kl, grad, hess = beta_kl(alpha1, beta1, true)

    ad_grad = ForwardDiff.gradient(beta_kl_wrapper, par)
    ad_hess = ForwardDiff.hessian(beta_kl_wrapper, par)

    # Check the derivatives
    @test_approx_eq beta_kl_wrapper(par) kl
    @test_approx_eq ad_grad grad
    @test_approx_eq ad_hess hess

    # Check the value
    p1_dist = Gamma(alpha1, beta1)
    p2_dist = Gamma(alpha2, beta2)
    test_kl_value(p1_dist, p2_dist, kl)
end


function test_categorical_kl_derivatives()
    p1 = Float64[ 1, 2, 3, 4]
    p2 = Float64[ 5, 6, 2, 1]

    p1 = p1 / sum(p1)
    p2 = p2 / sum(p2)

    categorical_kl = gen_categorical_kl(p2)

    function categorical_kl_wrapper{NumType <: Number}(par::Vector{NumType})
        return categorical_kl(par, false)
    end

    kl, grad, hess = categorical_kl(p1, true);

    ad_grad = ForwardDiff.gradient(categorical_kl_wrapper, p1)
    ad_hess = ForwardDiff.hessian(categorical_kl_wrapper, p1)

    # Check the derivatives
    @test_approx_eq categorical_kl_wrapper(p1) kl
    @test_approx_eq ad_grad grad
    @test_approx_eq ad_hess hess

    # Check the value
    p1_dist = Categorical(p1)
    p2_dist = Categorical(p2)
    test_kl_value(p1_dist, p2_dist, kl)
end


function test_diagmvn_mvn_kl_derivatives()
    K = 4
    mean1 = rand(K)
    var1 = rand(K)
    var1 = var1 .* var1

    mean2 = rand(K)
    cov2 = rand(K, K)
    cov2 = 0.2 * cov2 * cov2' + eye(K)

    diagmvn_mvn_kl = gen_diagmvn_mvn_kl(mu2, cov2)
    kl, grad_mean, grad_var, hess_mean, hess_var =
        diagmvn_mvn_kl(mean1, var1, true);

    hess = zeros(Float64, 2 * K, 2 * K)
    hess[1:K, 1:K] = hess_mean
    hess[(K + 1):(2 * K), (K + 1):(2 * K)] = hess_var

    function diagmvn_mvn_kl_wrapper{NumType <: Number}(par::Vector{NumType})
        K = Int(length(par) / 2)
        mean1 = par[1:K]
        var1 = par[(K + 1):(2 * K)]
        diagmvn_mvn_kl(mean1, var1, false)
    end

    par = vcat(mean1, var1)
    ad_grad = ForwardDiff.gradient(diagmvn_mvn_kl_wrapper, par)
    ad_hess = ForwardDiff.hessian(diagmvn_mvn_kl_wrapper, par)

    # Check the derivatives
    @test_approx_eq diagmvn_mvn_kl_wrapper(par) kl
    @test_approx_eq ad_grad vcat(grad_mean, grad_var)
    @test_approx_eq ad_hess hess

    # Check the value
    p1_dist = MvNormal(mean1, diagm(var1))
    p2_dist = MvNormal(mean2, cov2)
    test_kl_value(p1_dist, p2_dist, kl)
end


function test_normal_kl_derivatives()
    mean1 = 0.5
    var1 = 2.0

    mean2 = 0.8
    var2 = 1.8

    normal_kl = gen_normal_kl(mean2, var2)

    function normal_kl_wrapper{NumType <: Number}(par::Vector{NumType})
        normal_kl(par[1], par[2], false)
    end

    kl, grad, hess = normal_kl(mean1, var1, mean2, var2, true)

    par = vcat(mean1, var1)
    ad_grad = ForwardDiff.gradient(normal_kl_wrapper, par)
    ad_hess = ForwardDiff.hessian(normal_kl_wrapper, par)

    # Check the derivatives
    @test_approx_eq normal_kl_wrapper(par) kl
    @test_approx_eq ad_grad grad
    @test_approx_eq ad_hess hess

    # Check the value
    p1_dist = Normal(mean1, var1)
    p2_dist = Normal(mean2, var2)
    test_kl_value(p1_dist, p2_dist, kl)
end


###################################
# Run tests


@time test_set_hess()
@time test_real_image()
@time test_bvn_cov()
@time test_dual_numbers()
@time test_tile_predicted_image()
@time test_derivative_flags()
@time test_active_sources()
@time test_elbo()

# TODO: set in runtests
test_detailed_derivatives = false

if test_detailed_derivatives
    test_process_active_pixels()
    test_add_log_term()
    test_combine_pixel_sources()
    test_e_g_s_functions()
    test_fs1m_derivatives()
    test_fs0m_derivatives()
    test_bvn_derivatives()
    test_galaxy_variable_transform()
    test_galaxy_cache_component()
    test_galaxy_sigma_derivs()
    test_brightness_hessian()
    test_dsiginv_dsig()

    test_beta_kl_derivatives()
    test_categorical_kl_derivatives()
    test_diagmvn_mvn_kl_derivatives()
    test_normal_kl_derivatives()
end
