# Calculate values and partial derivatives of the variational ELBO.

module ElboDeriv

VERSION < v"0.4.0-dev" && using Docile
using CelesteTypes
import KL
import Util
import SloanDigitalSkySurvey: WCS
import WCSLIB

using DualNumbers.Dual

export tile_predicted_imagecom


####################################################
# Store pre-allocated memory in this data structures, which contains
# intermediate values used in the ELBO calculation.

type ElboIntermediateVariables{NumType <: Number}
  # Derivatives of a bvn with respect to (x, sig).
  bvn_x_d::Array{NumType, 1}
  bvn_sig_d::Array{NumType, 1}
  bvn_xx_h::Array{NumType, 2}
  bvn_xsig_h::Array{NumType, 2}
  bvn_sigsig_h::Array{NumType, 2}

  # intermediate values used in d bvn / d(x, sig)
  dpy1_dsig::Array{NumType, 1}
  dpy2_dsig::Array{NumType, 1}

  # TODO: delete this, it is now in BvnComponent
  dsiginv_dsig::Array{NumType, 2}

  # Derivatives of a bvn with respect to (u, shape)
  bvn_u_d::Array{NumType, 1}
  bvn_uu_h::Array{NumType, 2}
  bvn_s_d::Array{NumType, 1}
  bvn_ss_h::Array{NumType, 2}
  bvn_us_h::Array{NumType, 2}

  # Vectors of star and galaxy bvn quantities from all sources for a pixel.
  # The vector has one element for each active source, in the same order
  # as mp.active_sources.

  # TODO: you can treat this the same way as E_G_s and not keep a vector around.
  fs0m_vec::Vector{SensitiveFloat{StarPosParams, NumType}}
  fs1m_vec::Vector{SensitiveFloat{GalaxyPosParams, NumType}}

  # Brightness values for a single source
  E_G_s::SensitiveFloat{CanonicalParams, NumType}
  E_G2_s::SensitiveFloat{CanonicalParams, NumType}
  var_G_s::SensitiveFloat{CanonicalParams, NumType}

  # Expected pixel intensity and variance for a pixel from all sources.
  E_G::SensitiveFloat{CanonicalParams, NumType}
  var_G::SensitiveFloat{CanonicalParams, NumType}

  # A placeholder for the log term in the ELBO.
  elbo_log_term::SensitiveFloat{CanonicalParams, NumType}

  # The ELBO itself.
  elbo::SensitiveFloat{CanonicalParams, NumType}

  # A boolean.  If false, do not calculate hessians or derivatives.
  calculate_derivs::Bool
end


@doc """
Args:
  - S: The total number of sources
  - num_active_sources: The number of actives sources (with deriviatives)
""" ->
ElboIntermediateVariables(
    NumType::DataType, S::Int64, num_active_sources::Int64) = begin

  @assert NumType <: Number

  bvn_x_d = zeros(NumType, 2)
  bvn_sig_d = zeros(NumType, 3)
  bvn_xx_h = zeros(NumType, 2, 2)
  bvn_xsig_h = zeros(NumType, 2, 3)
  bvn_sigsig_h = zeros(NumType, 3, 3)

  dpy1_dsig = zeros(NumType, 3)
  dpy2_dsig = zeros(NumType, 3)
  dsiginv_dsig = zeros(NumType, 3, 3)

  # Derivatives wrt u.
  bvn_u_d = zeros(NumType, 2)
  bvn_uu_h = zeros(NumType, 2, 2)

  # Shape deriviatives.  Here, s stands for "shape".
  bvn_s_d = zeros(NumType, length(gal_shape_ids))

  # The hessians.
  bvn_ss_h = zeros(NumType, length(gal_shape_ids), length(gal_shape_ids))
  bvn_us_h = zeros(NumType, 2, length(gal_shape_ids))

  # fs0m and fs1m accumulate contributions from all bvn components
  # for a given source.
  fs0m_vec = Array(SensitiveFloat{StarPosParams, NumType}, S)
  fs1m_vec = Array(SensitiveFloat{GalaxyPosParams, NumType}, S)
  for s = 1:S
    fs0m_vec[s] = zero_sensitive_float(StarPosParams, NumType)
    fs1m_vec[s] = zero_sensitive_float(GalaxyPosParams, NumType)
  end

  E_G_s = zero_sensitive_float(CanonicalParams, NumType, 1)
  E_G2_s = zero_sensitive_float(CanonicalParams, NumType, 1)
  var_G_s = zero_sensitive_float(CanonicalParams, NumType, 1)

  E_G = zero_sensitive_float(CanonicalParams, NumType, num_active_sources)
  var_G = zero_sensitive_float(CanonicalParams, NumType, num_active_sources)

  elbo_log_term =
    zero_sensitive_float(CanonicalParams, NumType, num_active_sources)
  elbo = zero_sensitive_float(CanonicalParams, NumType, num_active_sources)

  ElboIntermediateVariables{NumType}(
    bvn_x_d, bvn_sig_d, bvn_xx_h, bvn_xsig_h, bvn_sigsig_h,
    dpy1_dsig, dpy2_dsig, dsiginv_dsig,
    bvn_u_d, bvn_uu_h, bvn_s_d, bvn_ss_h, bvn_us_h,
    fs0m_vec, fs1m_vec, E_G_s, E_G2_s, var_G_s,
    E_G, var_G, elbo_log_term, elbo, true)
end


include(joinpath(Pkg.dir("Celeste"), "src/ElboKL.jl"))
include(joinpath(Pkg.dir("Celeste"), "src/SourceBrightness.jl"))
include(joinpath(Pkg.dir("Celeste"), "src/BivariateNormals.jl"))

@doc """
Add the contributions of a star's bivariate normal term to the ELBO,
by updating elbo_vars.fs0m_vec[s] in place.

Args:
  - elbo_vars: Elbo intermediate values.
  - s: The index of the current source in 1:S
  - bmc: The component to be added
  - x: An offset for the component in pixel coordinates (e.g. a pixel location)
  - fs0m: A SensitiveFloat to which the value of the bvn likelihood
       and its derivatives with respect to x are added.
 - wcs_jacobian: The jacobian of the function pixel = F(world) at this location.
""" ->
function accum_star_pos!{NumType <: Number}(
    elbo_vars::ElboIntermediateVariables{NumType},
    s::Int64,
    bmc::BvnComponent{NumType},
    x::Vector{Float64},
    wcs_jacobian::Array{Float64, 2})

  py1, py2, f = eval_bvn_pdf(bmc, x)

  # TODO: Also make a version that doesn't calculate any derivatives
  # if the object isn't in active_sources.
  get_bvn_derivs!(elbo_vars, bmc, x, false);

  fs0m = elbo_vars.fs0m_vec[s]
  fs0m.v += f

  if elbo_vars.calculate_derivs
    # TODO: This wastes a _lot_ of calculation.  Make a version for
    # stars that only calculates the x derivatives.
    transform_bvn_derivs!(elbo_vars, bmc, wcs_jacobian)
    bvn_u_d = elbo_vars.bvn_u_d
    bvn_uu_h = elbo_vars.bvn_uu_h

    # Accumulate the derivatives.
    for u_id in 1:2
      fs0m.d[star_ids.u[u_id]] += f * bvn_u_d[u_id]
    end

    # Hessian terms involving only the location parameters.
    # TODO: redundant term
    for u_id1 in 1:2, u_id2 in 1:2
      fs0m.h[star_ids.u[u_id1], star_ids.u[u_id2]] +=
        f * (bvn_uu_h[u_id1, u_id2] + bvn_u_d[u_id1] * bvn_u_d[u_id2])
    end
  end
end


@doc """
Add the contributions of a galaxy component term to the ELBO by
updating fs1m in place.

Args:
  - elbo_vars: Elbo intermediate variables
  - s: The index of the current source in 1:S
  - gcc: The galaxy component to be added
  - x: An offset for the component in pixel coordinates (e.g. a pixel location)
  - wcs_jacobian: The jacobian of the function pixel = F(world) at this location.

Updates elbo_vars.fs1m_vec[sa] in place.
""" ->
function accum_galaxy_pos!{NumType <: Number}(
    elbo_vars::ElboIntermediateVariables{NumType},
    s::Int64,
    gcc::GalaxyCacheComponent{NumType},
    x::Vector{Float64},
    wcs_jacobian::Array{Float64, 2})

  py1, py2, f_pre = eval_bvn_pdf(gcc.bmc, x)
  f = f_pre * gcc.e_dev_i
  fs1m = elbo_vars.fs1m_vec[s];
  fs1m.v += f

  if elbo_vars.calculate_derivs

    get_bvn_derivs!(elbo_vars, gcc.bmc, x, true);
    transform_bvn_derivs!(elbo_vars, gcc, wcs_jacobian)

    bvn_u_d = elbo_vars.bvn_u_d
    bvn_uu_h = elbo_vars.bvn_uu_h
    bvn_s_d = elbo_vars.bvn_s_d
    bvn_ss_h = elbo_vars.bvn_ss_h
    bvn_us_h = elbo_vars.bvn_us_h

    # Accumulate the derivatives.
    for u_id in 1:2
      fs1m.d[gal_ids.u[u_id]] += f * bvn_u_d[u_id]
    end

    for gal_id in 1:length(gal_shape_ids)
      fs1m.d[gal_shape_alignment[gal_id]] += f * bvn_s_d[gal_id]
    end

    # The e_dev derivative.  e_dev just scales the entire component.
    # The direction is positive or negative depending on whether this
    # is an exp or dev component.
    fs1m.d[gal_ids.e_dev] += gcc.e_dev_dir * f_pre

    # The Hessians:

    # Hessian terms involving only the shape parameters.
    for shape_id1 in 1:length(gal_shape_ids), shape_id2 in 1:length(gal_shape_ids)
      s1 = gal_shape_alignment[shape_id1]
      s2 = gal_shape_alignment[shape_id2]
      fs1m.h[s1, s2] +=
        f * (bvn_ss_h[shape_id1, shape_id2] +
             bvn_s_d[shape_id1] * bvn_s_d[shape_id2])
    end

    # Hessian terms involving only the location parameters.
    for u_id1 in 1:2, u_id2 in 1:2
      u1 = gal_ids.u[u_id1]
      u2 = gal_ids.u[u_id2]
      fs1m.h[u1, u2] +=
        f * (bvn_uu_h[u_id1, u_id2] + bvn_u_d[u_id1] * bvn_u_d[u_id2])
    end

    # Hessian terms involving both the shape and location parameters.
    for u_id in 1:2, shape_id in 1:length(gal_shape_ids)
      ui = gal_ids.u[u_id]
      si = gal_shape_alignment[shape_id]
      fs1m.h[ui, si] +=
        f * (bvn_us_h[u_id, shape_id] + bvn_u_d[u_id] * bvn_s_d[shape_id])
      fs1m.h[si, ui] = fs1m.h[ui, si]
    end

    # Do the e_dev hessian terms.
    devi = gal_ids.e_dev
    for u_id in 1:2
      ui = gal_ids.u[u_id]
      fs1m.h[ui, devi] += f_pre * gcc.e_dev_dir * bvn_u_d[u_id]
      fs1m.h[devi, ui] = fs1m.h[ui, devi]
    end
    for shape_id in 1:length(gal_shape_ids)
      si = gal_shape_alignment[shape_id]
      fs1m.h[si, devi] += f_pre * gcc.e_dev_dir * bvn_s_d[shape_id]
      fs1m.h[devi, si] = fs1m.h[si, devi]
    end
  end # if calculate_derivs
end


@doc """
Populate fs0m_vec and fs1m_vec for all sources.
""" ->
function populate_fsm_vecs!{NumType <: Number}(
    elbo_vars::ElboIntermediateVariables{NumType},
    mp::ModelParams{NumType},
    tile::ImageTile,
    h::Int64, w::Int64,
    sbs::Vector{SourceBrightness{NumType}},
    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
    star_mcs::Array{BvnComponent{NumType}, 2})

  tile_sources = mp.tile_sources[tile.b][tile.hh, tile.ww]
  for s in tile_sources
    wcs_jacobian = mp.patches[s, tile.b].wcs_jacobian;
    sb = sbs[s];

    m_pos = Float64[tile.h_range[h], tile.w_range[w]]

    clear!(elbo_vars.fs0m_vec[s])
    for star_mc in star_mcs[:, s]
        accum_star_pos!(elbo_vars, s, star_mc, m_pos, wcs_jacobian)
    end

    clear!(elbo_vars.fs1m_vec[s])
    for i = 1:2 # Galaxy types
        for j in 1:[8,6][i] # Galaxy component
            for k = 1:3 # PSF component
                gal_mc = gal_mcs[k, j, i, s];
                accum_galaxy_pos!(elbo_vars, s, gal_mc, m_pos, wcs_jacobian)
            end
        end
    end
  end
end



@doc """
Add the contributions of a single source to E_G_s and var_G_s, which are cleared
and then updated in place.

Updates elbo_vars.E_G_s and elbo_vars.var_G_s in place.
""" ->
function accumulate_source_brightness!{NumType <: Number}(
    elbo_vars::ElboIntermediateVariables{NumType},
    mp::ModelParams{NumType},
    sbs::Vector{SourceBrightness{NumType}},
    s::Int64, b::Int64)

  # E[G] and E{G ^ 2} for a single source
  E_G = elbo_vars.E_G_s;
  E_G2 = elbo_vars.E_G2_s;

  clear!(E_G)
  clear!(E_G2)

  a = mp.vp[s][ids.a]
  fsm = (elbo_vars.fs0m_vec[s], elbo_vars.fs1m_vec[s]);
  sb = sbs[s];

  active_source = (s in mp.active_sources)

  for i in 1:Ia # Stars and galaxies
    lf = sb.E_l_a[b, i].v * fsm[i].v
    llff = sb.E_ll_a[b, i].v * fsm[i].v^2

    E_G.v += a[i] * lf
    E_G2.v += a[i] * llff

    # Only calculate derivatives for active sources.
    if active_source && elbo_vars.calculate_derivs
      ######################
      # Gradients.

      E_G.d[ids.a[i], 1] += lf
      E_G2.d[ids.a[i], 1] += llff

      p0_shape = shape_standard_alignment[i]
      p0_bright = brightness_standard_alignment[i]

      # Derivatives with respect to the spatial parameters
      a_fd = a[i] * fsm[i].d[:, 1]
      E_G.d[p0_shape, 1] += sb.E_l_a[b, i].v * a_fd
      E_G2.d[p0_shape, 1] += sb.E_ll_a[b, i].v * 2 * fsm[i].v * a_fd

      # Derivatives with respect to the brightness parameters.
      E_G.d[p0_bright, 1] +=
        a[i] * fsm[i].v * sb.E_l_a[b, i].d[:, 1]
      E_G2.d[p0_bright, 1] +=
        a[i] * (fsm[i].v^2) * sb.E_ll_a[b, i].d[:, 1]

      ######################
      # Hessians.

      # The (a, a) block of the hessian is zero.

      # The (bright, bright) block:

      # BLAS for
      # E_G.h[p0_bright, p0_bright] +=
      #   a[i] * fsm[i].v * sb.E_l_a[b, i].h
      # E_G2.h[p0_bright, p0_bright] +=
      #   a[i] * (fsm[i].v^2) * sb.E_ll_a[b, i].h
      # TODO: I think sub-indexing sb.E_l_a[b, i].h is allocating a lot of memory.
      BLAS.axpy!(a[i] * fsm[i].v,
                 sb.E_l_a[b, i].h, E_G.h[p0_bright, p0_bright]);
      BLAS.axpy!(a[i] * (fsm[i].v^2),
                 sb.E_ll_a[b, i].h, E_G2.h[p0_bright, p0_bright]);

      # The (shape, shape) block:
      # BLAS for
      # E_G.h[p0_shape, p0_shape] += a[i] * sb.E_l_a[b, i].v * fsm[i].h
      # E_G2.h[p0_shape, p0_shape] +=
      #   2 * a[i] * sb.E_ll_a[b, i].v *
      #   (fsm[i].v * fsm[i].h + fsm[i].d[:, 1] * fsm[i].d[:, 1]')
      BLAS.axpy!(a[i] * sb.E_l_a[b, i].v,
                 fsm[i].h, E_G.h[p0_shape, p0_shape]);
      BLAS.axpy!(2 * a[i] * sb.E_ll_a[b, i].v * fsm[i].v,
                 fsm[i].h, E_G2.h[p0_shape, p0_shape]);
      BLAS.ger!(2 * a[i] * sb.E_ll_a[b, i].v,
                fsm[i].d[:, 1], fsm[i].d[:, 1], E_G2.h[p0_shape, p0_shape]);

      # TODO: eliminate redundancy.
      # The (a, bright) blocks:
      h_a_bright = fsm[i].v * sb.E_l_a[b, i].d[:, 1]
      E_G.h[p0_bright, ids.a[i]] += h_a_bright
      E_G.h[ids.a[i], p0_bright] =  E_G.h[p0_bright, ids.a[i]]'

      h2_a_bright = (fsm[i].v ^ 2) * sb.E_ll_a[b, i].d[:, 1]
      E_G2.h[p0_bright, ids.a[i]] += h2_a_bright
      E_G2.h[ids.a[i], p0_bright] = E_G2.h[p0_bright, ids.a[i]]'

      # The (a, shape) blocks.
      h_a_shape = sb.E_l_a[b, i].v * fsm[i].d
      E_G.h[p0_shape, ids.a[i]] += h_a_shape
      E_G.h[ids.a[i], p0_shape] = E_G.h[p0_shape, ids.a[i]]'

      h2_a_shape = sb.E_ll_a[b, i].v * 2 * fsm[i].v * fsm[i].d[:, 1]
      E_G2.h[p0_shape, ids.a[i]] += h2_a_shape
      E_G2.h[ids.a[i], p0_shape] = E_G2.h[p0_shape, ids.a[i]]'

      # The (shape, bright) blocks.
      h_bright_shape = a[i] * sb.E_l_a[b, i].d[:, 1] * fsm[i].d'
      E_G.h[p0_bright, p0_shape] += h_bright_shape
      E_G.h[p0_shape, p0_bright] = E_G.h[p0_bright, p0_shape]'

      h2_bright_shape =
        2 * a[i] * sb.E_ll_a[b, i].d[:, 1] * fsm[i].v * fsm[i].d'
      E_G2.h[p0_bright, p0_shape] += h2_bright_shape
      E_G2.h[p0_shape, p0_bright] = E_G2.h[p0_bright, p0_shape]'
    end
  end

  # Write the variance as a function of (E_G, E_G2)
  clear!(elbo_vars.var_G_s)
  var_v = E_G2.v - (E_G.v ^ 2);

  if active_source && elbo_vars.calculate_derivs
    var_grad = NumType[-2 * E_G.v, 1];
    var_hess = NumType[-2  0; 0 0];
    combine_sfs!(E_G, E_G2, elbo_vars.var_G_s, var_v, var_grad, var_hess)
  else
    elbo_vars.var_G_s.v = var_v
  end
end


@doc """
Adds up E_G and var_G across all sources.

Updates elbo_vars.E_G and elbo_vars.var_G in place.
""" ->
function combine_pixel_sources!{NumType <: Number}(
    elbo_vars::ElboIntermediateVariables{NumType},
    mp::ModelParams{NumType},
    tile::ImageTile,
    sbs::Vector{SourceBrightness{NumType}})

  clear!(elbo_vars.E_G)
  clear!(elbo_vars.var_G)

  tile_sources = mp.tile_sources[tile.b][tile.hh, tile.ww];
  for s in tile_sources
    accumulate_source_brightness!(elbo_vars, mp, sbs, s, tile.b)
    add_sources_sf!(elbo_vars.E_G, elbo_vars.E_G_s, s)
    add_sources_sf!(elbo_vars.var_G, elbo_vars.var_G_s, s)
  end
end


@doc """
Expected pixel brightness.
Args:
  h: The row of the tile
  w: The column of the tile
  ...the rest are the same as elsewhere.
  tile_sources: The indices within active_sources that are present in the tile.

Returns:
  - Updates elbo_vars.E_G and elbo_vars.var_G in place.
""" ->
function get_expected_pixel_brightness!{NumType <: Number}(
    elbo_vars::ElboIntermediateVariables{NumType},
    h::Int64, w::Int64,
    sbs::Vector{SourceBrightness{NumType}},
    star_mcs::Array{BvnComponent{NumType}, 2},
    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
    tile::ImageTile,
    mp::ModelParams{NumType};
    include_epsilon::Bool=true)

  populate_fsm_vecs!(elbo_vars, mp, tile, h, w, sbs, gal_mcs, star_mcs)

  clear!(elbo_vars.E_G)
  clear!(elbo_vars.var_G)
  combine_pixel_sources!(elbo_vars, mp, tile, sbs);

  if include_epsilon
    # There are no derivatives with respect to epsilon, so can safely add
    # to the value.
    elbo_vars.E_G.v +=
      tile.constant_background ? tile.epsilon : tile.epsilon_mat[h, w]
  end
end


@doc """
Add the lower bound to the log term to the elbo for a single pixel.
As a side effect, elbo_vars.E_G2 is cleared.

Args:
   - elbo_vars: Intermediate variables
   - x_nbm: The photon count at this pixel
   - iota: The optical sensitivity

 Returns:
  Updates elbo_vars.elbo in place by adding the lower bound to the log
  term and clears E_G2.
""" ->
function add_elbo_log_term!{NumType <: Number}(
    elbo_vars::ElboIntermediateVariables{NumType},
    x_nbm::Float64, iota::Float64)

  # See notes for a derivation.  The log term is
  # log E[G] - Var(G) / (2 * E[G] ^2 )

  E_G = elbo_vars.E_G
  var_G = elbo_vars.var_G
  elbo = elbo_vars.elbo

  # The gradients and Hessians are written as a f(x, y) = f(E_G2, E_G)
  log_term_value = log(E_G.v) - 0.5 * var_G.v  / (E_G.v ^ 2)
  # println("Log term value: ", log_term_value)
  # println("E_G.v ", E_G.v)
  # println("var_G.v ", var_G.v)

  if elbo_vars.calculate_derivs
    log_term_grad = NumType[ -0.5 / (E_G.v ^ 2), 1 / E_G.v + var_G.v / (E_G.v ^ 3)]
    log_term_hess =
      NumType[0             1 / E_G.v^3;
              1 / E_G.v^3   -(1 / E_G.v ^ 2 + 3  * var_G.v / (E_G.v ^ 4))]

    # Desipte the variable name,
    # this step briefly updates E_G2 to contain the lower bound of the log term.
    CelesteTypes.combine_sfs!(
      elbo_vars.var_G, elbo_vars.E_G, elbo_vars.elbo_log_term,
      log_term_value, log_term_grad, log_term_hess)

    # Add to the elbo.
    add_value = elbo.v + x_nbm * (log(iota) + log_term_value)
    add_grad = NumType[1, x_nbm]
    add_hess = NumType[0 0; 0 0]
    CelesteTypes.combine_sfs!(
      elbo, elbo_vars.elbo_log_term, add_value, add_grad, add_hess)
  else
    # If not calculating derivatives, add the values directly.
    elbo.v += x_nbm * (log(iota) + log_term_value)
  end
end


############################################
# The remaining functions loop over tiles, sources, and pixels.

@doc """
Add a tile's contribution to the ELBO likelihood term by
modifying elbo in place.

Args:
  - tile: An image tile.
  - mp: The current model parameters.
  - sbs: The current source brightnesses.
  - star_mcs: All the star * PCF components.
  - gal_mcs: All the galaxy * PCF components.
  - elbo: The ELBO log likelihood to be updated.
""" ->
function tile_likelihood!{NumType <: Number}(
    elbo_vars::ElboIntermediateVariables{NumType},
    tile::ImageTile,
    mp::ModelParams{NumType},
    sbs::Vector{SourceBrightness{NumType}},
    star_mcs::Array{BvnComponent{NumType}, 2},
    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
    include_epsilon::Bool=true)

  elbo = elbo_vars.elbo
  tile_sources = mp.tile_sources[tile.b][tile.hh, tile.ww]

  # For speed, if there are no sources, add the noise
  # contribution directly.
  if (length(tile_sources) == 0) && include_epsilon
      # NB: not using the delta-method approximation here
      if tile.constant_background
          nan_pixels = Base.isnan(tile.pixels)
          num_pixels =
            length(tile.h_range) * length(tile.w_range) - sum(nan_pixels)
          tile_x = sum(tile.pixels[!nan_pixels])
          ep = tile.epsilon
          elbo.v += tile_x * log(ep) - num_pixels * ep
      else
          for w in 1:tile.w_width, h in 1:tile.h_width
              this_pixel = tile.pixels[h, w]
              if !Base.isnan(this_pixel)
                  ep = tile.epsilon_mat[h, w]
                  elbo.v += this_pixel * log(ep) - ep
              end
          end
      end
      return
  end

  # Iterate over pixels that are not NaN.
  for w in 1:tile.w_width, h in 1:tile.h_width
      this_pixel = tile.pixels[h, w]
      if !Base.isnan(this_pixel)
          get_expected_pixel_brightness!(
            elbo_vars, h, w, sbs, star_mcs, gal_mcs, tile,
            mp, include_epsilon=include_epsilon)
          iota = tile.constant_background ? tile.iota : tile.iota_vec[h]
          add_elbo_log_term!(elbo_vars, this_pixel, iota)
          CelesteTypes.add_scaled_sfs!(elbo_vars.elbo, elbo_vars.E_G, scale=-iota)
      end
  end

  # Subtract the log factorial term.  This is not a function of the
  # parameters so the derivatives don't need to be updated.
  elbo.v += -sum(lfact(tile.pixels[!Base.isnan(tile.pixels)]))
end


@doc """
Return the image predicted for the tile given the current parameters.

Args:
  - tile: An image tile.
  - mp: The current model parameters.
  - sbs: The current source brightnesses.
  - star_mcs: All the star * PCF components.
  - gal_mcs: All the galaxy * PCF components.
  - elbo: The ELBO log likelihood to be updated.

Returns:
  A matrix of the same size as the tile with the predicted brightnesses.
""" ->
function tile_predicted_image{NumType <: Number}(
        elbo_vars::ElboIntermediateVariables{NumType},
        tile::ImageTile,
        mp::ModelParams{NumType},
        sbs::Vector{SourceBrightness{NumType}},
        star_mcs::Array{BvnComponent{NumType}, 2},
        gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
        include_epsilon::Bool=true)

    predicted_pixels = copy(tile.pixels)
    # Iterate over pixels that are not NaN.
    for w in 1:tile.w_width, h in 1:tile.h_width
        this_pixel = tile.pixels[h, w]
        if !Base.isnan(this_pixel)
            get_expected_pixel_brightness!(
              elbo_vars, h, w, sbs, star_mcs, gal_mcs, tile,
              mp, include_epsilon=include_epsilon)
            iota = tile.constant_background ? tile.iota : tile.iota_vec[h]
            predicted_pixels[h, w] = E_G.v * iota
        end
    end

    predicted_pixels
end


@doc """
Produce a predicted image for a given tile and model parameters.

If include_epsilon is true, then the background is also rendered.
Otherwise, only pixels from the object are rendered.
""" ->
function tile_predicted_image{NumType <: Number}(
    tile::ImageTile, mp::ModelParams{NumType};
    include_epsilon::Bool=false)

  b = tile.b
  star_mcs, gal_mcs =
    load_bvn_mixtures(mp, b, calculate_derivs=false)
  sbs = SourceBrightness{NumType}[
    SourceBrightness(mp.vp[s], false) for s in 1:mp.S]

  elbo_vars = ElboIntermediateVariables(NumType, mp.S, length(mp.active_sources));
  elbo_vars.calculate_derivs = false

  tile_predicted_image(elbo_vars,
                       tile,
                       mp,
                       sbs,
                       star_mcs,
                       gal_mcs,
                       include_epsilon=include_epsilon)
end


@doc """
The ELBO likelihood for given brighntess and bvn components.
""" ->
function elbo_likelihood!{NumType <: Number}(
  elbo_vars::ElboIntermediateVariables{NumType},
  tiled_image::Array{ImageTile},
  mp::ModelParams{NumType},
  sbs::Vector{SourceBrightness{NumType}},
  star_mcs::Array{BvnComponent{NumType}, 2},
  gal_mcs::Array{GalaxyCacheComponent{NumType}, 4})

  @assert maximum(mp.active_sources) <= mp.S
  for tile in tiled_image[:]
    tile_sources = mp.tile_sources[tile.b][tile.hh, tile.ww]
    if length(intersect(tile_sources, mp.active_sources)) > 0
      tile_likelihood!(elbo_vars, tile, mp, sbs, star_mcs, gal_mcs);
    end
  end

end


@doc """
Add the expected log likelihood ELBO term for an image to elbo.

Args:
  - tiles: An array of ImageTiles
  - mp: The current model parameters.
  - elbo: A sensitive float containing the ELBO.
  - b: The current band
""" ->
function elbo_likelihood!{NumType <: Number}(
    elbo_vars::ElboIntermediateVariables{NumType},
    tiles::Array{ImageTile}, mp::ModelParams{NumType}, b::Int64,
    sbs::Vector{SourceBrightness{NumType}})

  star_mcs, gal_mcs =
    load_bvn_mixtures(mp, b, calculate_derivs=elbo_vars.calculate_derivs)
  elbo_likelihood!(elbo_vars, tiles, mp, sbs, star_mcs, gal_mcs)
end


@doc """
Return the expected log likelihood for all bands in a section
of the sky.
""" ->
function elbo_likelihood{NumType <: Number}(
    tiled_blob::TiledBlob, mp::ModelParams{NumType})

  elbo_vars = ElboIntermediateVariables(NumType, mp.S, length(mp.active_sources));
  sbs = load_source_brightnesses(mp, elbo_vars.calculate_derivs)
  for b in 1:length(tiled_blob)
      elbo_likelihood!(elbo_vars, tiled_blob[b], mp, b, sbs)
  end
  elbo_vars.elbo
end


@doc """
Calculates and returns the ELBO and its derivatives for all the bands
of an image.

Args:
  - tiled_blob: A TiledBlob.
  - mp: Model parameters.
""" ->
function elbo{NumType <: Number}(
    tiled_blob::TiledBlob, mp::ModelParams{NumType})
  elbo = elbo_likelihood(tiled_blob, mp)

  # TODO: subtract the kl with the hessian.
  subtract_kl!(mp, elbo)
  elbo
end



end
