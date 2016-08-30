import ForwardDiff

using Celeste: Model, SensitiveFloats, ElboDeriv
using Distributions
import ElboDeriv: BvnComponent, GalaxyCacheComponent
import ElboDeriv: eval_bvn_pdf!, get_bvn_derivs!, transform_bvn_derivs!


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


# TODO: fix this test.
function test_process_active_pixels()
    blob, ea, bodies = gen_two_body_dataset()

    # Choose four pixels only to keep the test fast.
    active_pixels = Array(ElboDeriv.ActivePixel, 4)
    active_pixels[1] = ElboDeriv.ActivePixel(1, 1, 10, 11)
    active_pixels[2] = ElboDeriv.ActivePixel(1, 1, 11, 10)
    active_pixels[3] = ElboDeriv.ActivePixel(5, 1, 10, 11)
    active_pixels[4] = ElboDeriv.ActivePixel(5, 1, 11, 10)


    function tile_lik_wrapper_fun{NumType <: Number}(
            ea::ElboArgs{NumType}, calculate_derivs::Bool)

        elbo_vars = ElboDeriv.ElboIntermediateVariables(
            NumType, ea.S,
            length(ea.active_sources),
            calculate_derivs=calculate_derivs,
            calculate_hessian=calculate_derivs)
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

    beta_kl = ElboDeriv.gen_beta_kl(alpha2, beta2)
    function beta_kl_wrapper{NumType <: Number}(par::Vector{NumType})
        alpha1 = par[1]
        beta1 = par[2]
        local kl, grad, hess
        kl, grad, hess = beta_kl(alpha1, beta1, false)
        return kl
    end

    kl, grad, hess = beta_kl(alpha1, beta1, true)

    ad_grad = ForwardDiff.gradient(beta_kl_wrapper, par)
    ad_hess = ForwardDiff.hessian(beta_kl_wrapper, par)

    # Check the derivatives
    @test_approx_eq beta_kl_wrapper(par) kl
    @test_approx_eq ad_grad grad
    @test_approx_eq ad_hess hess

    # Check the value
    p1_dist = Beta(alpha1, beta1)
    p2_dist = Beta(alpha2, beta2)
    test_kl_value(p1_dist, p2_dist, kl)
end


function test_categorical_kl_derivatives()
    p1 = Float64[ 1, 2, 3, 4]
    p2 = Float64[ 5, 6, 2, 1]

    p1 = p1 / sum(p1)
    p2 = p2 / sum(p2)

    categorical_kl = ElboDeriv.gen_categorical_kl(p2)

    function categorical_kl_wrapper{NumType <: Number}(par::Vector{NumType})
        local kl, grad, hess
        kl, grad, hess = categorical_kl(par, false)
        return kl
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
    cov2 = 0.5 * (cov2 + cov2')

    diagmvn_mvn_kl = ElboDeriv.gen_diagmvn_mvn_kl(mean2, cov2)
    kl, grad_mean, grad_var, hess_mean, hess_var =
        diagmvn_mvn_kl(mean1, var1, true);

    hess = zeros(Float64, 2 * K, 2 * K)
    hess[1:K, 1:K] = hess_mean
    hess[(K + 1):(2 * K), (K + 1):(2 * K)] = hess_var

    function diagmvn_mvn_kl_wrapper{NumType <: Number}(par::Vector{NumType})
        K = Int(length(par) / 2)
        mean1 = par[1:K]
        var1 = par[(K + 1):(2 * K)]
        local kl, grad_mean, grad_var, hess_mean, hess_var
        kl, grad_mean, grad_var, hess_mean, hess_var =
            diagmvn_mvn_kl(mean1, var1, false)
        return kl
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

    normal_kl = ElboDeriv.gen_normal_kl(mean2, var2)

    function normal_kl_wrapper{NumType <: Number}(par::Vector{NumType})
        local kl, grad, hess
        kl, grad, hess = normal_kl(par[1], par[2], false)
        return kl
    end

    kl, grad, hess = normal_kl(mean1, var1, true)

    par = vcat(mean1, var1)
    ad_grad = ForwardDiff.gradient(normal_kl_wrapper, par)
    ad_hess = ForwardDiff.hessian(normal_kl_wrapper, par)

    # Check the derivatives
    @test_approx_eq normal_kl_wrapper(par) kl
    @test_approx_eq ad_grad grad
    @test_approx_eq ad_hess hess

    # Check the value
    p1_dist = Normal(mean1, sqrt(var1))
    p2_dist = Normal(mean2, sqrt(var2))
    test_kl_value(p1_dist, p2_dist, kl)
end




########################
# Test derivatives of sensitive float manipulations

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
  combine_sfs!(sf1, sf2, sf1.v[1] ^ 2 * sqrt(sf2.v[1]), g_d, g_h);

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

  add_sources_sf!(sf_all, sf_s, 1, true)

  x2 = rand(P);
  clear!(sf_s);
  scaled_exp!(sf_s, x2, a2);
  v2 = sf_s.v

  fd_grad2 = ForwardDiff.gradient(f2, x2);
  @test_approx_eq sf_s.d fd_grad2

  fd_hess2 = ForwardDiff.hessian(f2, x2);
  @test_approx_eq sf_s.h fd_hess2

  add_sources_sf!(sf_all, sf_s, 2, true)

  @test_approx_eq (v1 + v2) sf_all.v

  @test_approx_eq (v1 + v2) sf_all.v
  @test_approx_eq fd_grad1 sf_all.d[1:P, 1]
  @test_approx_eq fd_grad2 sf_all.d[1:P, 2]
  @test_approx_eq fd_hess1 sf_all.h[1:P, 1:P]
  @test_approx_eq fd_hess2 sf_all.h[(1:P) + P, (1:P) + P]
  @test_approx_eq zeros(P, P) sf_all.h[(1:P), (1:P) + P]
  @test_approx_eq zeros(P, P) sf_all.h[(1:P) + P, (1:P)]
end


#####################################
# Transforms

function test_box_derivatives()
	blob, ea, body = gen_three_body_dataset();
	transform = Transform.get_mp_transform(ea, loc_width=1.0);

	box_params = setdiff(fieldnames(ids), [:a, :k])
	vp_free = transform.from_vp(ea.vp)
	for sa = 1:length(ea.active_sources), param in box_params, ind in length(getfield(ids, param))
		# sa = 1
		# param = box_params[1]
		# ind = 1

		s = ea.active_sources[sa]
		vp_ind = getfield(ids, param)[ind]
		free_ind = [getfield(ids_free, param)[ind]]

		function wrap_transform{NumType <: Number}(vp_free_s::Vector{NumType})
			local_vp_free =
				Array{NumType, 1}[ convert(Array{NumType, 1}, vp_free[sa]) for
			                  sa = 1:length(ea.active_sources) ]
			local_vp_free[s] = vp_free_s
			vp = transform.to_vp(local_vp_free)
			vp[s][getfield(ids, param)[ind]]
		end

		ad_d  = ForwardDiff.gradient(wrap_transform, vp_free[s])[free_ind][1]
		ad_h = ForwardDiff.hessian(wrap_transform, vp_free[s])[free_ind, free_ind][1,1]

		d, h = Transform.box_derivatives(
			ea.vp[s][vp_ind][1], transform.bounds[s][param][ind])
		@test_approx_eq ad_d d
		@test_approx_eq ad_h h
	end
end


function test_box_simplex_derivatives()
	blob, ea, body = gen_three_body_dataset();
	for s = 1:ea.S
		delta = 0.01 * s # Make the parameters different for each one
		ea.vp[s][ids.a] = Float64[ 0.2 - delta, 0.8 + delta ]
		ea.vp[s][ids.k] = Float64[ 0.2- delta 0.2- delta; 0.8 + delta 0.8 + delta ]
	end
	transform = Transform.get_mp_transform(ea, loc_width=1.0);

	simplex_params = [:a, :k]
	vp_free = transform.from_vp(ea.vp)

	for sa = 1:length(ea.active_sources), param in simplex_params
		# sa = 1
		# param = :k
		# col = 1 # For k only
		# ind = 1 # Index within the simplex

		s = ea.active_sources[sa]
		num_cols = length(size(getfield(ids, param)))
		@assert num_cols == 1 || num_cols == 2

		for col = 1:num_cols
			vp_ind = getfield(ids, param)[:, col]

			if length(size(getfield(ids_free, param))) == 0
				# Hack to handle ids_free.a
				@assert col == 1
				free_ind = [ getfield(ids_free, param) ]
			else
				free_ind = getfield(ids_free, param)[:, col]
			end

			d, h = Transform.box_simplex_derivatives(
				ea.vp[s][vp_ind], transform.bounds[s][param][col])

			for row = 1:2
				# Write with a univariate output so we can take autodiff hessians.
			  function wrap_transform{NumType <: Number}(vp_free_s::Vector{NumType})
			  	local_vp_free =
			  		Array{NumType, 1}[ convert(Array{NumType, 1}, vp_free[sa]) for
			  	                     sa = 1:length(ea.active_sources) ]
			  	local_vp_free[s] = vp_free_s
			  	vp = transform.to_vp(local_vp_free)
			  	vp[s][getfield(ids, param)[row, col]]
			  end

			  ad_d = ForwardDiff.gradient(wrap_transform, vp_free[s])[free_ind]
			  ad_h = ForwardDiff.hessian(wrap_transform, vp_free[s])[free_ind, free_ind]
				@test_approx_eq ad_d d[row, :][1]
				@test_approx_eq ad_h h[row]
			end
		end
	end
end


function test_simplex_derivatives()
	n = 4
	basic_simplex_box = Transform.SimplexBox(0, 1, n)
	z = Float64[1, 2, 4, 3]
	z /= sum(z)
	r = Transform.unsimplexify_parameter(z, basic_simplex_box)
	Transform.simplexify_parameter(r, basic_simplex_box)

	ad_d = Array(Array{Float64}, n)
	ad_h = Array(Array{Float64}, n)

	for ind = 1:n
	  function wrap_simplex{NumType <: Number}(r::Vector{NumType})
	    local z = Transform.simplexify_parameter(r, basic_simplex_box)
	    z[ind]
	  end
	  ad_d[ind] = ForwardDiff.gradient(wrap_simplex, r)
	  ad_h[ind] = ForwardDiff.hessian(wrap_simplex, r)
	end

	jacobian, hessian_vec = Transform.simplex_derivatives(z)

	@test_approx_eq jacobian' reduce(hcat, ad_d)
	for ind = 1:n
		@test_approx_eq(hessian_vec[ind], ad_h[ind])
	end
end


###################################
# Run tests

# ELBO tests:
println("Running ELBO derivative tests.")
@time test_combine_pixel_sources()
@time test_fs1m_derivatives()
@time test_fs0m_derivatives()
@time test_bvn_derivatives()
@time test_galaxy_variable_transform()
@time test_galaxy_cache_component()
@time test_galaxy_sigma_derivs()
@time test_brightness_hessian()
@time test_dsiginv_dsig()
@time test_add_log_term()
@time test_e_g_s_functions()
#test_process_active_pixels()  # TODO: fix this

# KL tests:
println("Running KL derivative tests.")
@time test_beta_kl_derivatives()
@time test_categorical_kl_derivatives()
@time test_diagmvn_mvn_kl_derivatives()
@time test_normal_kl_derivatives()

# SensitiveFloat tests:
println("Running SensitiveFloat derivative tests.")
@time test_combine_sfs()
@time test_add_sources_sf()

# Transform tests:
println("Running Transform derivative tests.")
@time test_box_derivatives()
@time test_box_simplex_derivatives()
@time test_simplex_derivatives()
