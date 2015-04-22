# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

# Calculate values and partial derivatives of the variational ELBO.

module ElboDeriv

using CelesteTypes
import Util

immutable SourceBrightness
    # SensitiveFloat objects for expectations involving r_s and c_s.
    #
    # Args:
    #   vs: A vector of variational parameters
    #
    # Attributes:
    #   Each matrix has one row for each color and a column for
    #   star / galaxy.  Row 3 is the gamma distribute baseline brightness,
    #   and all other rows are lognormal offsets.
    #   E_l_a: A B x Ia matrix of expectations and derivatives of
    #     color terms.  The rows are bands, and the columns
    #     are star / galaxy.
    #   E_ll_a: A B x Ia matrix of expectations and derivatives of
    #     squared color terms.  The rows are bands, and the columns
    #     are star / galaxy.

    E_l_a::Matrix{SensitiveFloat}  # [E[l|a=0], E[l]|a=1]]
    E_ll_a::Matrix{SensitiveFloat}   # [E[l^2|a=0], E[l^2]|a=1]]

    SourceBrightness(vs::Vector{Float64}) = begin
        gamma_s = vs[ids.gamma]
        zeta = vs[ids.zeta]
        beta = vs[ids.beta]
        lambda = vs[ids.lambda]

        # E_l_a has a row for each of the five colors and columns
        # for star / galaxy.
        E_l_a = Array(SensitiveFloat, B, Ia)

        for i = 1:Ia
            for b = 1:B
                E_l_a[b, i] = zero_sensitive_float([-1], all_params)
            end

            # Index 3 is r_s and has a gamma expectation.
            E_l_a[3, i].v = gamma_s[i] * zeta[i]
            E_l_a[3, i].d[ids.gamma[i]] = zeta[i]
            E_l_a[3, i].d[ids.zeta[i]] = gamma_s[i]

            # The remaining indices involve c_s and have lognormal
            # expectations times E_c_3.
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

        E_ll_a = Array(SensitiveFloat, B, Ia)
        for i = 1:Ia
            for b = 1:B
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
    # Relevant parameters of a bivariate normal distribution.
    #
    # Args:
    #   the_mean: The mean as a 2x1 column vector
    #   the_cov: The covaraiance as a 2x2 matrix
    #   weight: A scalar weight
    #
    # Attributes:
    #    the_mean: The mean argument
    #    precision: The inverse of the_cov
    #    z: The weight times the normalizing constant.

    the_mean::Vector{Float64}
    precision::Matrix{Float64}
    z::Float64

    BvnComponent(the_mean, the_cov, weight) = begin
        the_det = the_cov[1,1] * the_cov[2,2] - the_cov[1,2] * the_cov[2,1]
        c = 1 ./ (the_det^.5 * 2pi)
        new(the_mean, the_cov^-1, c * weight)
    end
end


immutable GalaxyCacheComponent
    # A the convolution of a one galaxy component with one PSF component.
    #
    # Args:
    #  - theta_dir: "Theta direction": this is 1 or -1, depending on whether
    #      increasing theta increases the weight of this GalaxyCacheComponent
    #      (1) or decreases it (-1).
    #  - theta_i: The weight given to this type of galaxy for this celestial object.
    #      This is either theta or (1 - theta).
    #  - gc: The galaxy component to be convolved
    #  - pc: The psf component to be convolved
    #  - mu: The location of the celestial object as a 2x1 vector
    #  - rho: The ratio of the galaxy minor axis to major axis (0 < rho <= 1)
    #  - sigma: The scale of the galaxy major axis
    #
    # Attributes:
    #  - theta_dir: Same as input
    #  - theta_i: Same as input
    #  - bmc: A BvnComponent with the convolution.
    #  - dSigma: A 3x3 matrix containing the derivates of
    #      [Sigma11, Sigma12, Sigma22] (in the rows) with respect to
    #      [rho, phi, sigma]

    theta_dir::Float64
    theta_i::Float64
    bmc::BvnComponent
    dSigma::Matrix{Float64}  # [Sigma11, Sigma12, Sigma22] x [rho, phi, sigma]

    GalaxyCacheComponent(theta_dir::Float64, theta_i::Float64,
            gc::GalaxyComponent, pc::PsfComponent, mu::Vector{Float64},
            rho::Float64, phi::Float64, sigma::Float64) = begin
        XiXi = Util.get_bvn_cov(rho, phi, sigma)
        mean_s = [pc.xiBar[1] + mu[1], pc.xiBar[2] + mu[2]]
        var_s = pc.SigmaBar + gc.sigmaTilde * XiXi
        weight = pc.alphaBar * gc.alphaTilde  # excludes theta
        bmc = BvnComponent(mean_s, var_s, weight)

        dSigma = Array(Float64, 3, 3)
        cos_sin = cos(phi)sin(phi)
        sin_sq = sin(phi)^2
        cos_sq = cos(phi)^2
        dSigma[:, 1] = 2rho * sigma^2 * [sin_sq, -cos_sin, cos_sq]
        dSigma[:, 2] = sigma^2 * (rho^2 - 1) * [2cos_sin, sin_sq - cos_sq, -2cos_sin]
        dSigma[:, 3] = (2XiXi ./ sigma)[[1, 2, 4]]
        dSigma .*= gc.sigmaTilde

        new(theta_dir, theta_i, bmc, dSigma)
    end
end


function load_bvn_mixtures(psf::Vector{PsfComponent}, mp::ModelParams)
    # Convolve the current locations and galaxy shapes with the PSF.
    #
    # Args:
    #  - psf: A vector of PSF components
    #  - mp: The current ModelParams
    #
    # Returns:
    #  - star_mcs: An # of PSF components x # of sources array of BvnComponents
    #  - gal_mcs: An array of BvnComponents with indices
    #     - PSF component
    #     - Galaxy component
    #     - Galaxy type
    #     - Source

    # The PSF contains three components, so you see lots of 3's below.
    # TODO: don't hard-code the number of PSF components
    # TODO: don't hard code the number of galaxy components (8 and 6 below) either.

    star_mcs = Array(BvnComponent, 3, mp.S)
    gal_mcs = Array(GalaxyCacheComponent, 3, 8, 2, mp.S)

    for s in 1:mp.S
        vs = mp.vp[s]

        # Convolve the star locations with the PSF.
        for k in 1:3
            pc = psf[k]
            mean_s = [pc.xiBar[1] + vs[ids.mu[1]], pc.xiBar[2] + vs[ids.mu[2]]]
            star_mcs[k, s] = BvnComponent(mean_s, pc.SigmaBar, pc.alphaBar)
        end

        # Convolve the galaxy representations with the PSF.
        for i = 1:Ia
            # TODO: Jeff, could you say what theta_dir is?
            theta_dir = (i == 1) ? 1. : -1.
            theta_i = (i == 1) ? vs[ids.theta] : 1. - vs[ids.theta]

            # Galaxies of type 1 have 8 components, and type 2 have 6 components (?)
            for j in 1:[8,6][i]
                for k = 1:3
                    gal_mcs[k, j, i, s] = GalaxyCacheComponent(
                        theta_dir, theta_i, galaxy_prototypes[i][j], psf[k],
                        vs[ids.mu], vs[ids.rho], vs[ids.phi], vs[ids.sigma])
                end
            end
        end
    end

    star_mcs, gal_mcs
end


function ret_pdf(bmc::BvnComponent, x::Vector{Float64})
    # Return quantities related to the pdf of an offset bivariate normal.
    #
    # Args:
    #   - bmc: A bivariate normal component
    #   - x: A 2x1 vector containing a mean offset to be applied to bmc
    #
    # Returns:
    #   - Stuff

    y1 = x[1] - bmc.the_mean[1]
    y2 = x[2] - bmc.the_mean[2]
    py1 = bmc.precision[1,1] * y1 + bmc.precision[1,2] * y2
    py2 = bmc.precision[2,1] * y1 + bmc.precision[2,2] * y2
    c_ytpy = -0.5 * (y1 * py1 + y2 * py2)
    f_denorm = exp(c_ytpy)
    py1, py2, bmc.z * f_denorm
end


function accum_star_pos!(bmc::BvnComponent,
                         x::Vector{Float64},
                         fs0m::SensitiveFloat)
    # Add the contributions of a star's bivariate normal term to the ELBO.
    #
    # Args:
    #   - bmc: The component to be added
    #   - x: An offset for the component (e.g. a pixel location)
    #   - fs0m: A SensitiveFloat to which the value of the bvn likelihood
    #        and its derivatives with respect to x are added.
    #
    # Returns:
    #   Updates fs0m in place.

    py1, py2, f = ret_pdf(bmc, x)

    fs0m.v += f

    # TODO: reference this with ids
    fs0m.d[1] += f .* py1 #mu1
    fs0m.d[2] += f .* py2 #mu2
end


function accum_galaxy_pos!(gcc::GalaxyCacheComponent,
                           x::Vector{Float64},
                           fs1m::SensitiveFloat)
    # Add the contributions of a galaxy component term to the ELBO.
    #
    # Args:
    #   - gcc: The galaxy component to be added
    #   - x: An offset for the component (e.g. a pixel location)
    #   - fs1m: A SensitiveFloat to which the value of the likelihood
    #        and its derivatives with respect to x are added.
    #
    # Returns:
    #   Updates fs1m in place.

    py1, py2, f_pre = ret_pdf(gcc.bmc, x)
    f = f_pre * gcc.theta_i

    fs1m.v += f

    # TODO: reference this with ids
    fs1m.d[1] += f .* py1 #mu1
    fs1m.d[2] += f .* py2 #mu2
    fs1m.d[3] += gcc.theta_dir * f_pre #theta

    df_dSigma = Array(Float64, 3)
    df_dSigma[1] = 0.5 * f * (py1 * py1 - gcc.bmc.precision[1, 1])
    df_dSigma[2] = f * (py1 * py2 - gcc.bmc.precision[1, 2])  # NB: 2X
    df_dSigma[3] = 0.5 * f * (py2 * py2 - gcc.bmc.precision[2, 2])

    for i in 1:3  # [drho, dphi, dsigma]
        for j in 1:3  # [dSigma11, dSigma12, dSigma22]
            fs1m.d[i + 3] += df_dSigma[j] * gcc.dSigma[j, i]
        end
    end
end


function accum_pixel_source_stats!(sb::SourceBrightness,
        star_mcs::Array{BvnComponent, 2},
        gal_mcs::Array{GalaxyCacheComponent, 4},
        vs::Vector{Float64}, child_s::Int64, parent_s::Int64,
        m_pos::Vector{Float64}, b::Int64,
        fs0m::SensitiveFloat, fs1m::SensitiveFloat,
        E_G::SensitiveFloat, var_G::SensitiveFloat)
    # Add up the ELBO values and derivatives for a single source
    # in a single band.
    #
    # Args:
    #   - sb: The source's brightness expectations and derivatives
    #   - star_mcs: An array of star * PSF components.  The index
    #       order is PSF component x source.
    #   - gal_mcs: An array of galaxy * PSF components.  The index order is
    #       PSF component x galaxy component x galaxy type x source
    #   - vs: The variational parameters for this source
    #   - child_s: The index of this source within the tile.
    #   - parent_s: The global index of this source.
    #   - m_pos: A 2x1 vector with the pixel location
    #   - b: The band (1 to 5)
    #   - fs0m: The accumulated star contributions (updated in place)
    #   - fs1m: The accumulated galaxy contributions (updated in place)
    #   - E_G: Expected celestial signal in this band (G_{nbm})
    #        (updated in place)
    #   - var_G: Variance of G (updated in place)
    #
    # Returns:
    #   - Clears and updates fs0m, fs1m with the total
    #     star and galaxy contributions to the ELBO from this source
    #     in this band.  Adds the contributions to E_G and var_G.

    # Accumulate over PSF components.
    clear!(fs0m)
    for star_mc in star_mcs[:, parent_s]
        accum_star_pos!(star_mc, m_pos, fs0m)
    end

    clear!(fs1m)
    # TODO: Don't hard-code the index ranges.
    for i = 1:2 # Galaxy types
        for j in 1:[8,6][i] # Galaxy component
            for k = 1:3 # PSF component
                accum_galaxy_pos!(gal_mcs[k, j, i, parent_s], m_pos, fs1m)
            end
        end
    end

    # Add the contributions of this source in this band to
    # E(G) and Var(G).

    # In the structures below, 1 = star and 2 = galaxy.
    chi = vs[ids.chi]
    fsm = (fs0m, fs1m)
    lf = (sb.E_l_a[b, 1].v * fs0m.v, sb.E_l_a[b, 2].v * fs1m.v)
    llff = (sb.E_ll_a[b, 1].v * fs0m.v^2, sb.E_ll_a[b, 2].v * fs1m.v^2)

    E_G_s_v = chi[1] * lf[1] + chi[2] * lf[2]
    E_G.v += E_G_s_v

    # These formulas for the variance of G use the fact that the
    # variational distributions of each source and band are independent.
    var_G.v -= E_G_s_v^2
    var_G.v += chi[1] * llff[1] + chi[2] * llff[2]

    # Add the contributions of this source in this band to
    # the derivatibes of E(G) and Var(G).

    # Chi derivatives:
    E_G.d[ids.chi[1], child_s] += lf[1]
    E_G.d[ids.chi[2], child_s] += lf[2]

    var_G.d[ids.chi[1], child_s] -= 2 * E_G_s_v * lf[1]
    var_G.d[ids.chi[2], child_s] -= 2 * E_G_s_v * lf[2]

    var_G.d[ids.chi[1], child_s] += llff[1]
    var_G.d[ids.chi[2], child_s] += llff[2]

    # Derivatives with respect to the normal component parameters.
    for i in 1:Ia # Stars and galaxies
        # Loop over parameters for each fsm component.
        for p1 in 1:length(fsm[i].param_index)
            p0 = fsm[i].param_index[p1]
            chi_fd = chi[i] * fsm[i].d[p1]
            chi_El_fd = sb.E_l_a[b, i].v * chi_fd
            E_G.d[p0, child_s] += chi_El_fd
            var_G.d[p0, child_s] -= 2 * E_G_s_v * chi_El_fd
            var_G.d[p0, child_s] += chi_fd * sb.E_ll_a[b, i].v * 2 * fsm[i].v
        end
    end

    # Derivatives with respect to the brightness parameters.
    for i in 1:Ia # Stars and galaxies
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
    # Add the contributions of the expected value of a G term to the ELBO.
    #
    # Args:
    #   - tile_sources: A vector of source ids influencing this tile
    #   - x_nbm: The photon count at this pixel
    #   - iota: The optical sensitivity
    #   - E_G: The variational expected value of G
    #   - var_G: The variational variance of G
    #   - ret: A SensitiveFloat for the ELBO which is updated
    #
    # Returns:
    #   - Adds the contributions of E_G and var_G to ret in place.


    # Accumulate the values.
    # Add the lower bound to the E_q[log(F_{nbm})] term
    ret.v += x_nbm * (log(iota) + log(E_G.v) - var_G.v / (2. * E_G.v^2))

    # Subtract the E_q[F_{nbm}] term.
    ret.v -= iota * E_G.v

    # Accumulate the derivatives.
    for child_s in 1:length(tile_sources), p in 1:size(E_G.d, 1)
        parent_s = tile_sources[child_s]

        # Derivative of the log term lower bound.
        ret.d[p, parent_s] +=
            x_nbm * (E_G.d[p, child_s] / E_G.v
                     - 0.5 * (E_G.v^2 * var_G.d[p, child_s]
                              - var_G.v * 2 * E_G.v * E_G.d[p, child_s])
                        ./  E_G.v^4)

        # Derivative of the linear term.
        ret.d[p, parent_s] -= iota * E_G.d[p, child_s]
    end
end


function tile_range(tile::ImageTile, tile_width::Int64)
    # Return the range of image pixels in an ImageTile.

    h1 = 1 + (tile.hh - 1) * tile_width
    h2 = min(tile.hh * tile_width, tile.img.H)
    w1 = 1 + (tile.ww - 1) * tile_width
    w2 = min(tile.ww * tile_width, tile.img.W)
    h1:h2, w1:w2
end


function local_sources(tile::ImageTile, mp::ModelParams)
    # Return 
    #
    # Args:
    #   - tile: An ImageTile (containing tile coordinates)
    #   - mp: Model parameters.
    #
    # Returns:
    #   - A vector of source ids (from 1 to mp.S) that influence
    #     pixels in the tile.  A source influences a tile if
    #     there is any overlap in their squares of influence.

    local_subset = Array(Int64, 0)

    # "Radius" is used in the sense of an L_{\infty} norm.
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
        star_mcs::Array{BvnComponent, 2},
        gal_mcs::Array{GalaxyCacheComponent, 4},
        accum::SensitiveFloat)
    # Add a tile's contribution to the ELBO likelihood term.
    #
    # Args:
    #   - tile: An image tile.
    #   - mp: The current model parameters.
    #   - sbs: The currne source brightnesses.
    #   - star_mcs: All the star * PCF components.
    #   - gal_mcs: All the galaxy * PCF components.
    #   - accum: The ELBO log likelihood to be updated.
    #
    # Returns:
    #   - Adds the tile's contributions to the ELBO log likelihood
    #     to accum in place. 

    tile_sources = local_sources(tile, mp)
    h_range, w_range = tile_range(tile, mp.tile_width)

    # For speed, if there are no sources, add the noise
    # contribution directly.
    if length(tile_sources) == 0
        num_pixels = length(h_range) * length(w_range)
        tile_x = sum(tile.img.pixels[h_range, w_range])
        ep = tile.img.epsilon
        # NB: not using the delta-method approximation here
        accum.v += tile_x * log(ep) - num_pixels * ep
        return
    end

    # fs0m and fs1m accumulate contributions from all sources,
    # and so we say their derivatives are with respect to
    # source "-1".
    fs0m = zero_sensitive_float([-1], star_pos_params)
    fs1m = zero_sensitive_float([-1], galaxy_pos_params)

    E_G = zero_sensitive_float(tile_sources, all_params)
    var_G = zero_sensitive_float(tile_sources, all_params)

    # Iterate over pixels.
    for w in w_range, h in h_range
        clear!(E_G)
        E_G.v = tile.img.epsilon
        clear!(var_G)

        m_pos = Float64[h, w]
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
    # Add the expected log likelihood ELBO term for an image to accum.
    #
    # Args:
    #   - img: An image
    #   - mp: The current model parameters.
    #   - accum: A sensitive float containing the ELBO.
    #
    # Returns:
    #   - Adds the expected log likelihood to accum in place.

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
    # Return the expected log likelihood for all bands in a section
    # of the sky.

    ret = zero_sensitive_float([1:mp.S], all_params)
    for img in blob
        elbo_likelihood!(img, mp, ret)
    end
    ret
end


function subtract_kl_c!(d::Int64, i::Int64, s::Int64,
                        mp::ModelParams,
                        accum::SensitiveFloat)
    # Subtract from accum the entropy and expected prior of
    # the variational distribution of c
    # for a source (s), a source type (i, star / galaxy), and
    # color prior component (d).
    #
    # Args:
    #   - d: A color prior component.
    #   - i: Source type (star / galaxy).
    #   - s: Source id.
    #   - mp: Model parameters.
    #   - accum: The ELBO value.
    #
    # Returns:
    #   Updates accum in place.

    vs = mp.vp[s]

    # The below entropy / prior expectation are of
    # (c | a_i, kappa), and the final contribution needs to
    # be weighted by chi (the probability of this celestial object
    # type) and kappa (the probability of this color prior component).

    chi_si = vs[ids.chi[i]]
    half_kappa = .5 * vs[ids.kappa[d, i]]

    beta, lambda = (vs[ids.beta[:, i]], vs[ids.lambda[:, i]])
    Omega, Lambda = (mp.pp.Omega[i][:, d], mp.pp.Lambda[i][d])

    diff = Omega - beta
    Lambda_inv = Lambda^-1  # TODO: cache this!

    # NB: In the below expressions the variational entropy
    # and expected log prior are kind of mixed up together. 
    ret = sum(diag(Lambda_inv) .* lambda) - 4
    ret += (diff' * Lambda_inv * diff)[]
    ret += -sum(log(lambda)) + logdet(Lambda)
    accum.v -= chi_si * ret * half_kappa

    # Accumulate derivatives.
    accum.d[ids.kappa[d, i], s] -= chi_si * .5 * ret
    accum.d[ids.beta[:, i], s] -= chi_si * half_kappa * 2Lambda_inv * -diff
    accum.d[ids.lambda[:, i], s] -= chi_si * half_kappa * diag(Lambda_inv)
    accum.d[ids.lambda[:, i], s] -= chi_si * half_kappa ./ -lambda
    accum.d[ids.chi[i], s] -= ret * half_kappa
end


function subtract_kl_k!(i::Int64, s::Int64,
                        mp::ModelParams,
                        accum::SensitiveFloat)
    # Subtract from accum the entropy and expected prior of
    # the variational distribution of k
    # for a source (s), and source type (i, star / galaxy).
    #
    # Args:
    #   - i: Source type (star / galaxy).
    #   - s: Source id.
    #   - mp: Model parameters.
    #   - accum: The ELBO value.
    #
    # Returns:
    #   Updates accum in place.

    vs = mp.vp[s]

    chi_si = vs[ids.chi[i]]
    kappa_i = vs[ids.kappa[:, i]]

    for d in 1:D
        log_ratio = log(kappa_i[d] / mp.pp.Xi[i][d])
        kappa_log_ratio = kappa_i[d] * log_ratio
        accum.v -= chi_si * kappa_log_ratio
        accum.d[ids.kappa[d, i] , s] -= chi_si * (1 + log_ratio)
        accum.d[ids.chi[i], s] -= kappa_log_ratio
    end
end


function subtract_kl_r!(i::Int64, s::Int64,
                        mp::ModelParams, accum::SensitiveFloat)
    # Subtract from accum the entropy and expected prior of
    # the variational distribution of r
    # for a source (s), and source type (i, star / galaxy).
    #
    # Args:
    #   - i: Source type (star / galaxy).
    #   - s: Source id.
    #   - mp: Model parameters.
    #   - accum: The ELBO value.
    #
    # Returns:
    #   Updates accum in place.

    vs = mp.vp[s]
    gamma_si = mp.vp[s][ids.gamma[i]]
    zeta_si = mp.vp[s][ids.zeta[i]]

    digamma_gamma = digamma(gamma_si)
    zeta_Psi_ratio = (zeta_si - mp.pp.Psi[i]) / mp.pp.Psi[i]
    shape_diff = gamma_si - mp.pp.Upsilon[i]

    kl_v = shape_diff * digamma_gamma
    kl_v += -lgamma(gamma_si) + lgamma(mp.pp.Upsilon[i])
    kl_v += mp.pp.Upsilon[i] * (log(mp.pp.Psi[i]) - log(zeta_si))
    kl_v += gamma_si * zeta_Psi_ratio

    chi_si = vs[ids.chi[i]]
    accum.v -= chi_si * kl_v

    accum.d[ids.gamma[i], s] -= chi_si * shape_diff * polygamma(1, gamma_si)
    accum.d[ids.gamma[i], s] -= chi_si * zeta_Psi_ratio

    accum.d[ids.zeta[i], s] -= chi_si * (-mp.pp.Upsilon[i] / zeta_si)
    accum.d[ids.zeta[i], s] -= chi_si * (gamma_si / mp.pp.Psi[i])

    accum.d[ids.chi[i], s] -= kl_v
end


function subtract_kl_a!(s::Int64, mp::ModelParams, accum::SensitiveFloat)
    # Subtract from accum the entropy and expected prior of
    # the variational distribution of a source (s).
    #
    # Args:
    #   - s: Source id.
    #   - mp: Model parameters.
    #   - accum: The ELBO value.
    #
    # Returns:
    #   Updates accum in place.

    Phi = mp.pp.Phi

    for i in 1:Ia
        chi_s = mp.vp[s][ids.chi[i]]
        accum.v -= chi_s * (log(chi_s) - log(Phi))
        accum.d[ids.chi[i], s] -= (log(chi_s) - log(Phi)) + 1
    end
end


function subtract_kl!(mp::ModelParams, accum::SensitiveFloat)
    # Subtract from accum the entropy and expected prior of
    # the variational distribution.

    for s in 1:mp.S
        subtract_kl_a!(s, mp, accum)

        # TODO: Do not hard-code constants.
        for i in 1:Ia
            subtract_kl_r!(i, s, mp, accum)
            subtract_kl_k!(i, s, mp, accum)
            for d in 1:D
                subtract_kl_c!(d, i, s, mp, accum)
            end
        end
    end
end


function elbo(blob::Blob, mp::ModelParams)
    # Caculate the ELBO for all the bands of an image.
    #
    # Args:
    #   - blob: An image.
    #   - mp: Model parameters.
    #
    # Returns:
    #   - The ELBO and its derivatives.

    ret = elbo_likelihood(blob, mp)
    subtract_kl!(mp, ret)
    ret
end

end

