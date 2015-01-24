# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module ElboDeriv

using CelesteTypes
import Util


immutable SourceBrightness
    E_l_a::Matrix{SensitiveFloat}  # [E[l|a=0], E[l]|a=1]]
    E_ll_a::Matrix{SensitiveFloat}   # [E[l^2|a=0], E[l^2]|a=1]]

    SourceBrightness(vs::Vector{Float64}) = begin
        chi = vs[ids.chi]
        gamma_s = vs[ids.gamma]
        zeta = vs[ids.zeta]
        beta = vs[ids.beta]
        lambda = vs[ids.lambda]

        E_l_a = Array(SensitiveFloat, 5, 2)

        for i = 1:2
            for b = 1:5
                E_l_a[b, i] = zero_sensitive_float([-1], all_params)
            end

            E_l_a[3, i].v = gamma_s[i] * zeta[i]
            E_l_a[3, i].d[ids.gamma[i]] = zeta[i]
            E_l_a[3, i].d[ids.zeta[i]] = gamma_s[i]

            E_c_3 = exp(beta[3, i] + .5 * lambda[3, i])
            E_l_a[4, i].v = E_l_a[3, i].v * E_c_3
            E_l_a[4, i].d[ids.gamma[i]] = E_l_a[3, i].d[ids.gamma[i]] * E_c_3
            E_l_a[4, i].d[ids.zeta[i]] = E_l_a[3, i].d[ids.zeta[i]] * E_c_3
            E_l_a[4, i].d[ids.beta[3, i]] = E_l_a[4, i].v
            E_l_a[4, i].d[ids.lambda[3, i]] = E_l_a[4, i].v * .5

            E_c_4 = exp(beta[4, i] + .5 * lambda[4, i])
            E_l_a[5, i].v = E_l_a[4, i].v * E_c_4
            E_l_a[5, i].d[ids.gamma[i]] = E_l_a[4, i].d[ids.gamma[i]] * E_c_4
            E_l_a[5, i].d[ids.zeta[i]] = E_l_a[4, i].d[ids.zeta[i]] * E_c_4
            E_l_a[5, i].d[ids.beta[3, i]] = E_l_a[4, i].d[ids.beta[3, i]] * E_c_4
            E_l_a[5, i].d[ids.lambda[3, i]] = E_l_a[4, i].d[ids.lambda[3, i]] * E_c_4
            E_l_a[5, i].d[ids.beta[4, i]] = E_l_a[5, i].v
            E_l_a[5, i].d[ids.lambda[4, i]] = E_l_a[5, i].v * .5

            E_c_2 = exp(-beta[2, i] + .5 * lambda[2, i])
            E_l_a[2, i].v = E_l_a[3, i].v * E_c_2
            E_l_a[2, i].d[ids.gamma[i]] = E_l_a[3, i].d[ids.gamma[i]] * E_c_2
            E_l_a[2, i].d[ids.zeta[i]] = E_l_a[3, i].d[ids.zeta[i]] * E_c_2
            E_l_a[2, i].d[ids.beta[2, i]] = E_l_a[2, i].v * -1.
            E_l_a[2, i].d[ids.lambda[2, i]] = E_l_a[2, i].v * .5

            E_c_1 = exp(-beta[1, i] + .5 * lambda[1, i])
            E_l_a[1, i].v = E_l_a[2, i].v * E_c_1
            E_l_a[1, i].d[ids.gamma[i]] = E_l_a[2, i].d[ids.gamma[i]] * E_c_1
            E_l_a[1, i].d[ids.zeta[i]] = E_l_a[2, i].d[ids.zeta[i]] * E_c_1
            E_l_a[1, i].d[ids.beta[2, i]] = E_l_a[2, i].d[ids.beta[2, i]] * E_c_1
            E_l_a[1, i].d[ids.lambda[2, i]] = E_l_a[2, i].d[ids.lambda[2, i]] * E_c_1
            E_l_a[1, i].d[ids.beta[1, i]] = E_l_a[1, i].v * -1.
            E_l_a[1, i].d[ids.lambda[1, i]] = E_l_a[1, i].v * .5 
        end

        E_ll_a = Array(SensitiveFloat, 5, 2)
        for i = 1:2
            for b = 1:5
                E_ll_a[b, i] = zero_sensitive_float([-1], all_params)
            end

            zeta_sq = zeta[i]^2
            E_ll_a[3, i].v = gamma_s[i] * (1 + gamma_s[i]) * zeta_sq
            E_ll_a[3, i].d[ids.gamma[i]] = (1 + 2 * gamma_s[i]) * zeta_sq
            E_ll_a[3, i].d[ids.zeta[i]] = 2 * gamma_s[i] * (1. + gamma_s[i]) * zeta[i]

            tmp3 = exp(2beta[3, i] + 2 * lambda[3, i])
            E_ll_a[4, i].v = E_ll_a[3, i].v * tmp3
            E_ll_a[4, i].d[:] = E_ll_a[3, i].d * tmp3
            E_ll_a[4, i].d[ids.beta[3, i]] = E_ll_a[4, i].v * 2.
            E_ll_a[4, i].d[ids.lambda[3, i]] = E_ll_a[4, i].v * 2. 

            tmp4 = exp(2beta[4, i] + 2 * lambda[4, i])
            E_ll_a[5, i].v = E_ll_a[4, i].v * tmp4
            E_ll_a[5, i].d[:] = E_ll_a[4, i].d * tmp4
            E_ll_a[5, i].d[ids.beta[4, i]] = E_ll_a[5, i].v * 2.
            E_ll_a[5, i].d[ids.lambda[4, i]] = E_ll_a[5, i].v * 2.

            tmp2 = exp(-2beta[2, i] + 2 * lambda[2, i])
            E_ll_a[2, i].v = E_ll_a[3, i].v * tmp2
            E_ll_a[2, i].d[:] = E_ll_a[3, i].d * tmp2
            E_ll_a[2, i].d[ids.beta[2, i]] = E_ll_a[2, i].v * -2.
            E_ll_a[2, i].d[ids.lambda[2, i]] = E_ll_a[2, i].v * 2. 

            tmp1 = exp(-2beta[1, i] + 2 * lambda[1, i])
            E_ll_a[1, i].v = E_ll_a[2, i].v * tmp1
            E_ll_a[1, i].d[:] = E_ll_a[2, i].d * tmp1
            E_ll_a[1, i].d[ids.beta[1, i]] = E_ll_a[1, i].v * -2.
            E_ll_a[1, i].d[ids.lambda[1, i]] = E_ll_a[1, i].v * 2. 
        end

        new(E_l_a, E_ll_a)
    end
end


immutable BvnComponent
    the_mean::Vector{Float64}
    precision::Matrix{Float64}
    z::Float64

    BvnComponent(the_mean, the_cov, weight) = begin
        the_det = the_cov[1,1] * the_cov[2,2] - the_cov[1,2] * the_cov[2,1]
        c = 1 ./ (the_det^.5 * 2pi)
        new(the_mean, the_cov^-1, c * weight)
    end
end


function load_bvn_mixtures(psf::Vector{PsfComponent}, mp::ModelParams)
    star_mcs = Array(BvnComponent, 3, mp.S)
    gal_mcs = Array(BvnComponent, 3, 8, 2, mp.S)

    for s in 1:mp.S
        vs = mp.vp[s]

        for k in 1:3
            pc = psf[k]
            mean_s = [pc.xiBar[1] + vs[ids.mu[1]], pc.xiBar[2] + vs[ids.mu[2]]]
            star_mcs[k, s] = BvnComponent(mean_s, pc.SigmaBar, pc.alphaBar)
        end

        Xi = [[vs[ids.Xi[1]] vs[ids.Xi[2]]], [0. vs[ids.Xi[3]]]]
        XiXi = Xi' * Xi

        for i = 1:2
            for j in 1:[6,8][i]
                gc = galaxy_prototypes[i][j]
                for k = 1:3
                    pc = psf[k]
                    mean_s = [pc.xiBar[1] + vs[ids.mu[1]], pc.xiBar[2] + vs[ids.mu[2]]]
                    var_s = pc.SigmaBar + gc.sigmaTilde * XiXi
                    weight = pc.alphaBar * gc.alphaTilde
                    gal_mcs[k, j, i, s] = BvnComponent(mean_s, var_s, weight)
                end
            end
        end
    end

    star_mcs, gal_mcs
end


function ret_pdf(bmc::BvnComponent, x::Vector{Float64})
    y1 = x[1] - bmc.the_mean[1]
    y2 = x[2] - bmc.the_mean[2]
    py1 = bmc.precision[1,1] * y1 + bmc.precision[1,2] * y2
    py2 = bmc.precision[2,1] * y1 + bmc.precision[2,2] * y2
    c_ytpy = -0.5 * (y1 * py1 + y2 * py2)
    f_denorm = exp(c_ytpy)
    py1, py2, bmc.z * f_denorm
end


function accum_star_pos!(bmc::BvnComponent, x::Vector{Float64},
        fs0m::SensitiveFloat)
    py1, py2, f = ret_pdf(bmc, x)

    fs0m.v += f
    fs0m.d[1] += f .* py1 #mu1
    fs0m.d[2] += f .* py2 #mu2
end


function accum_galaxy_pos!(bmc::BvnComponent, x::Vector{Float64},
        theta_i::Float64, theta_dir::Float64, st::Float64,
        Xi::Vector{Float64}, fs1m::SensitiveFloat)
    py1, py2, f_pre = ret_pdf(bmc, x)
    f = f_pre * theta_i

    fs1m.v += f
    fs1m.d[1] += f .* py1 #mu1
    fs1m.d[2] += f .* py2 #mu2
    fs1m.d[3] += theta_dir * f_pre #theta

    df_dSigma_11 = 0.5 * f * (py1 * py1 - bmc.precision[1, 1])
    df_dSigma_12 = f * (py1 * py2 - bmc.precision[1, 2])  # NB: 2X
    df_dSigma_22 = 0.5 * f * (py2 * py2 - bmc.precision[2, 2])

    fs1m.d[4] += st * (df_dSigma_11 * 2Xi[1] + df_dSigma_12 * Xi[2])
    fs1m.d[5] += st * (df_dSigma_12 * Xi[1] + df_dSigma_22 * 2Xi[2])
    fs1m.d[6] += st * (df_dSigma_22 * 2Xi[3])
end


function accum_pixel_source_stats!(sb::SourceBrightness,
        star_mcs::Array{BvnComponent, 2}, gal_mcs::Array{BvnComponent, 4},
        vs::Vector{Float64}, child_s::Int64, parent_s,
        m_pos::Vector{Float64}, b::Int64,
        fs0m::SensitiveFloat, fs1m::SensitiveFloat, 
        E_G::SensitiveFloat, var_G::SensitiveFloat)

    clear!(fs0m)
    for star_mc in star_mcs[:, parent_s]
        accum_star_pos!(star_mc, m_pos, fs0m)
    end

    clear!(fs1m)
    for i = 1:2
        theta_dir = (i == 1) ? 1. : -1.
        theta_i = (i == 1) ? vs[ids.theta] : 1. - vs[ids.theta]

        for j in 1:[6,8][i]
            for k = 1:3
                accum_galaxy_pos!(gal_mcs[k, j, i, parent_s], m_pos, theta_i, 
                    theta_dir, galaxy_prototypes[i][j].sigmaTilde, vs[ids.Xi], fs1m)
            end
        end
    end

    chi = (1. - vs[ids.chi], vs[ids.chi])
    fsm = (fs0m, fs1m)
    lf = (sb.E_l_a[b, 1].v * fs0m.v, sb.E_l_a[b, 2].v * fs1m.v)
    llff = (sb.E_ll_a[b, 1].v * fs0m.v^2, sb.E_ll_a[b, 2].v * fs1m.v^2)

    E_G_s_v = chi[1] * lf[1] + chi[2] * lf[2]
    E_G.v += E_G_s_v
    var_G.v -= E_G_s_v^2
    var_G.v += chi[1] * llff[1] + chi[2] * llff[2]

    lf_diff = lf[2] - lf[1]
    E_G.d[ids.chi, child_s] += lf_diff
    var_G.d[ids.chi, child_s] -= 2 * E_G_s_v * lf_diff
    var_G.d[ids.chi, child_s] += llff[2] - llff[1]
    for i in 1:2
        for p1 in 1:length(fsm[i].param_index)
            p0 = fsm[i].param_index[p1]
            chi_fd = chi[i] * fsm[i].d[p1]
            chi_El_fd = sb.E_l_a[b, i].v * chi_fd
            E_G.d[p0, child_s] += chi_El_fd
            var_G.d[p0, child_s] -= 2 * E_G_s_v * chi_El_fd
            var_G.d[p0, child_s] += chi_fd * sb.E_ll_a[b, i].v * 2 * fsm[i].v
        end
    end
    
    for i in 1:2
        for p0 in vcat(ids.gamma, ids.zeta, ids.beta[:], ids.lambda[:])
            chi_f_Eld = chi[i] * fsm[i].v * sb.E_l_a[b, i].d[p0]
            E_G.d[p0, child_s] += chi_f_Eld
            var_G.d[p0, child_s] -= 2 * E_G_s_v * chi_f_Eld
            var_G.d[p0, child_s] += chi[i] * fsm[i].v^2 * sb.E_ll_a[b, i].d[p0]
        end
    end
end


function accum_pixel_ret!(tile_sources::Vector{Int64}, 
        x_nbm::Float64, iota::Float64,
        E_G::SensitiveFloat, var_G::SensitiveFloat, ret::SensitiveFloat)

    ret.v += x_nbm * (log(iota) + log(E_G.v) - var_G.v / (2. * E_G.v^2))
    ret.v -= iota * E_G.v

    for child_s in 1:length(tile_sources), p in 1:size(E_G.d, 1)
        parent_s = tile_sources[child_s]
        ret.d[p, parent_s] += x_nbm * (E_G.d[p, child_s] / E_G.v
            - 0.5 * (E_G.v^2 * var_G.d[p, child_s] - 
                var_G.v * 2 * E_G.v * E_G.d[p, child_s]) 
                    ./  E_G.v^4)
        ret.d[p, parent_s] -= iota * E_G.d[p, child_s]
    end
end


function tile_range(tile::ImageTile, tile_width::Int64)
    h1 = 1 + (tile.hh - 1) * tile_width 
    h2 = min(tile.hh * tile_width, tile.img.H)
    w1 = 1 + (tile.ww - 1) * tile_width 
    w2 = min(tile.ww * tile_width, tile.img.W)
    h1:h2, w1:w2
end


function local_sources(tile::ImageTile, mp::ModelParams)
    local_subset = Array(Int64, 0)

    tr = mp.tile_width / 2.  # tile radius
    tc1 = tr + (tile.hh - 1) * mp.tile_width
    tc2 = tr + (tile.ww - 1) * mp.tile_width

    for s in 1:mp.S
        pc = mp.patches[s].center  # patch center
        pr = mp.patches[s].radius  # patch radius

        if abs(pc[1] - tc1) <= (pr + tr) && abs(pc[2] - tc2) <= (pr + tr)
            push!(local_subset, s)
        end
    end

    local_subset
end

function elbo_likelihood!(tile::ImageTile, mp::ModelParams, 
        sbs::Vector{SourceBrightness},
         star_mcs::Array{BvnComponent, 2}, gal_mcs::Array{BvnComponent, 4}, 
        accum::SensitiveFloat)
    tile_sources = local_sources(tile, mp)
    h_range, w_range = tile_range(tile, mp.tile_width)

    if length(tile_sources) == 0  # special case---for speed
        num_pixels = length(h_range) * length(w_range)
        tile_x = sum(tile.img.pixels[h_range, w_range])
        ep = tile.img.epsilon
        accum.v += tile_x * log(ep) - num_pixels * ep
        return
    end

    fs0m = zero_sensitive_float([-1], star_pos_params)
    fs1m = zero_sensitive_float([-1], galaxy_pos_params)

    E_G = zero_sensitive_float(tile_sources, all_params)
    var_G = zero_sensitive_float(tile_sources, all_params)

    for w in w_range, h in h_range
        clear!(E_G)  #serious bottleneck
        E_G.v = tile.img.epsilon
        clear!(var_G)

        m_pos = Float64[h - 0.5, w - 0.5]
        for child_s in 1:length(tile_sources)
            parent_s = tile_sources[child_s]
            accum_pixel_source_stats!(sbs[parent_s], star_mcs, gal_mcs, 
                mp.vp[parent_s], child_s, parent_s, m_pos, tile.img.b, 
                fs0m, fs1m, E_G, var_G)
        end
        
        accum_pixel_ret!(tile_sources, tile.img.pixels[h, w], tile.img.iota,
            E_G, var_G, accum)
    end
end


function elbo_likelihood!(img::Image, mp::ModelParams, accum::SensitiveFloat)
    accum.v += -sum(lfact(img.pixels))

    star_mcs, gal_mcs = load_bvn_mixtures(img.psf, mp)

    sbs = [SourceBrightness(mp.vp[s]) for s in 1:mp.S]

    WW = int(ceil(img.W / mp.tile_width))
    HH = int(ceil(img.H / mp.tile_width))
    for ww in 1:WW, hh in 1:HH
        tile = ImageTile(hh, ww, img)
        # might get a speedup from subsetting the mp here
        elbo_likelihood!(tile, mp, sbs, star_mcs, gal_mcs, accum)
    end
end


function elbo_likelihood(blob::Blob, mp::ModelParams)
    ret = zero_sensitive_float([1:mp.S], all_params)
    for img in blob
        elbo_likelihood!(img, mp, ret)
    end
    ret
end


function subtract_kl_c!(d::Int64, i::Int64, s::Int64, mp::ModelParams, 
        accum::SensitiveFloat)
    vs = mp.vp[s]
    beta, lambda = (vs[ids.beta[:, i]], vs[ids.lambda[:, i]])
    Omega, Lambda = (mp.pp.Omega[i][:, d], mp.pp.Lambda[i][d])

    diff = Omega - beta
    Lambda_inv = Lambda^-1  # cache this!
    half_kappa = .5 * vs[ids.kappa[d, i]]

    ret = sum(diag(Lambda_inv) .* lambda) - 4
    ret += (diff' * Lambda_inv * diff)[]
    ret += -sum(log(lambda)) + logdet(Lambda)
    accum.v -= ret * half_kappa

    accum.d[ids.kappa[d, i], s] -= .5 * ret
    accum.d[ids.beta[:, i], s] -= half_kappa * 2Lambda_inv * -diff
    accum.d[ids.lambda[:, i], s] -= half_kappa * diag(Lambda_inv)
    accum.d[ids.lambda[:, i], s] -= half_kappa ./ -lambda
end


function subtract_kl_k!(i::Int64, s::Int64, mp::ModelParams, accum::SensitiveFloat)
    kappa = mp.vp[s][ids.kappa[:, i]]
    for d in 1:D
        log_ratio = log(kappa[d] / mp.pp.Psi[i][d])
        accum.v -= kappa[d] * log_ratio
        accum.d[ids.kappa[d, i] , s] -= 1 + log_ratio
    end
end


function subtract_kl_r!(i::Int64, s::Int64, mp::ModelParams, accum::SensitiveFloat)
    vs = mp.vp[s]
    gamma_si = mp.vp[s][ids.gamma[i]]
    zeta_si = mp.vp[s][ids.zeta[i]]

    digamma_gamma = digamma(gamma_si)
    zeta_Phi_ratio = (zeta_si - mp.pp.Phi[i]) / mp.pp.Phi[i]
    shape_diff = gamma_si - mp.pp.Upsilon[i]

    accum.v -= shape_diff * digamma_gamma
    accum.v -= -lgamma(gamma_si) + lgamma(mp.pp.Upsilon[i])
    accum.v -= mp.pp.Upsilon[i] * (log(mp.pp.Phi[i]) - log(zeta_si))
    accum.v -= gamma_si * zeta_Phi_ratio

    accum.d[ids.gamma[i], s] -= shape_diff * polygamma(1, gamma_si)
    accum.d[ids.gamma[i], s] -= zeta_Phi_ratio

    accum.d[ids.zeta[i], s] -= -mp.pp.Upsilon[i] / zeta_si
    accum.d[ids.zeta[i], s] -= gamma_si / mp.pp.Phi[i]
end


function subtract_kl_a!(s::Int64, mp::ModelParams, accum::SensitiveFloat)
    chi_s = mp.vp[s][ids.chi]
    Delta = mp.pp.Delta

    accum.v -= chi_s * (log(chi_s) - log(Delta))
    accum.v -= (1. - chi_s) * (log(1. - chi_s) - log(1. - Delta))

    accum.d[ids.chi, s] -= (log(chi_s) - log(Delta)) + 1
    accum.d[ids.chi, s] -= -(log(1. - chi_s) - log(1. - Delta)) - 1.
end


function subtract_kl!(mp::ModelParams, accum::SensitiveFloat)
    for s in 1:mp.S
        subtract_kl_a!(s, mp, accum)
        for i in 1:2
            subtract_kl_r!(i, s, mp, accum)
            subtract_kl_k!(i, s, mp, accum)
            for d in 1:D
                subtract_kl_c!(d, i, s, mp, accum)
            end
        end
    end
end


function elbo(blob::Blob, mp::ModelParams)
    ret = elbo_likelihood(blob, mp)
    subtract_kl!(mp, ret)
    ret
end


end

