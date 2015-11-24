# Calculate values and partial derivatives of the variational ELBO.

module ElboDeriv

VERSION < v"0.4.0-dev" && using Docile
using CelesteTypes
import KL
import Util
import Polygons
import SloanDigitalSkySurvey: WCS
import WCSLIB

using DualNumbers.Dual

export tile_predicted_image
export ParameterMessage, update_parameter_message!

include(joinpath(Pkg.dir("Celeste"), "src/ElboKL.jl"))
include(joinpath(Pkg.dir("Celeste"), "src/SourceBrightness.jl"))
include(joinpath(Pkg.dir("Celeste"), "src/BivariateNormals.jl"))


@doc """
Add the contributions of a star's bivariate normal term to the ELBO,
by updating fs0m in place.

Args:
  - bmc: The component to be added
  - x: An offset for the component in pixel coordinates (e.g. a pixel location)
  - fs0m: A SensitiveFloat to which the value of the bvn likelihood
       and its derivatives with respect to x are added.
 - wcs_jacobian: The jacobian of the function pixel = F(world) at this location.
""" ->
function accum_star_pos!{NumType <: Number}(
                         bmc::BvnComponent{NumType},
                         x::Vector{Float64},
                         fs0m::SensitiveFloat{StarPosParams, NumType},
                         wcs_jacobian::Array{Float64, 2})
    py1, py2, f = eval_bvn_pdf(bmc, x)

    # TODO: This wastes a _lot_ of calculation.  Make a version for
    # stars that only calculates the x derivatives.
    bvn_sf = get_bvn_derivs(bmc, x);
    bvn_u_d, bvn_uu_h = transform_bvn_derivs(bvn_sf, bmc, wcs_jacobian)

    fs0m.v += f

    # Accumulate the derivatives.
    for u_id in 1:2
      fs0m.d[star_ids.u[u_id]] += f * bvn_u_d[u_id]
    end

    # Hessian terms involving only the location parameters.
    # TODO: redundant term
    for u_id1 in 1:2, u_id2 in 1:2
      fs0m.hs[1][star_ids.u[u_id1], star_ids.u[u_id2]] +=
        f * (bvn_uu_h[u_id1, u_id2] + bvn_u_d[u_id1] * bvn_u_d[u_id2])
    end
end


@doc """
Add the contributions of a galaxy component term to the ELBO by
updating fs1m in place.

Args:
  - gcc: The galaxy component to be added
  - x: An offset for the component in pixel coordinates (e.g. a pixel location)
  - fs1m: A SensitiveFloat to which the value of the likelihood
       and its derivatives with respect to x are added.
 - wcs_jacobian: The jacobian of the function pixel = F(world) at this location.
""" ->
function accum_galaxy_pos!{NumType <: Number}(
                           gcc::GalaxyCacheComponent{NumType},
                           x::Vector{Float64},
                           fs1m::SensitiveFloat{GalaxyPosParams, NumType},
                           wcs_jacobian::Array{Float64, 2})
    py1, py2, f_pre = eval_bvn_pdf(gcc.bmc, x)
    f = f_pre * gcc.e_dev_i

    fs1m.v += f

    bvn_sf = get_bvn_derivs(gcc.bmc, x);
    bvn_u_d, bvn_s_d, bvn_uu_h, bvn_ss_h, bvn_us_h =
      transform_bvn_derivs(bvn_sf, gcc, wcs_jacobian)

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
      fs1m.hs[1][s1, s2] +=
        f * (bvn_ss_h[shape_id1, shape_id2] +
             bvn_s_d[shape_id1] * bvn_s_d[shape_id2])
    end

    # Hessian terms involving only the location parameters.
    for u_id1 in 1:2, u_id2 in 1:2
      u1 = gal_ids.u[u_id1]
      u2 = gal_ids.u[u_id2]
      fs1m.hs[1][u1, u2] +=
        f * (bvn_uu_h[u_id1, u_id2] + bvn_u_d[u_id1] * bvn_u_d[u_id2])
    end

    # Hessian terms involving both the shape and locaiton parameters.
    for u_id in 1:2, shape_id in 1:length(gal_shape_ids)
      ui = gal_ids.u[u_id]
      si = gal_shape_alignment[shape_id]
      fs1m.hs[1][ui, si] +=
        f * (bvn_us_h[u_id, shape_id] + bvn_u_d[u_id] * bvn_s_d[shape_id])
      fs1m.hs[1][si, ui] = fs1m.hs[1][ui, si]

      # Do the e_dev hessians while we're here.
      devi = gal_ids.e_dev
      fs1m.hs[1][ui, devi] = f_pre * gcc.e_dev_dir * bvn_u_d[u_id]
      fs1m.hs[1][devi, ui] = fs1m.hs[1][ui, devi]
      fs1m.hs[1][si, devi] = f_pre * gcc.e_dev_dir * bvn_s_d[shape_id]
      fs1m.hs[1][devi, si] = fs1m.hs[1][si, devi]
    end
end


@doc """
Add up the ELBO values and derivatives for a single source
in a single band.

Args:
  - sb: The source's brightness expectations and derivatives
  - star_mcs: An array of star * PSF components.  The index
      order is PSF component x source.
  - gal_mcs: An array of galaxy * PSF components.  The index order is
      PSF component x galaxy component x galaxy type x source
  - vs: The variational parameters for this source
  - child_s: The index of this source within the tile.
  - parent_s: The global index of this source.
  - m_pos: A 2x1 vector with the pixel location in pixel coordinates
  - b: The band (1 to 5)
  - fs0m: The accumulated star contributions (updated in place)
  - fs1m: The accumulated galaxy contributions (updated in place)
  - E_G: Expected celestial signal in this band (G_{nbm})
       (updated in place)
  - var_G: Variance of G (updated in place)
  - wcs_jacobian: The jacobian of the function pixel = F(world) at this location.

Returns:
  - Clears and updates fs0m, fs1m with the total
    star and galaxy contributions to the ELBO from this source
    in this band.  Adds the contributions to E_G and var_G.
""" ->
function accum_pixel_source_stats!{NumType <: Number}(
        sb::SourceBrightness{NumType},
        star_mcs::Array{BvnComponent{NumType}, 2},
        gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
        vs::Vector{NumType}, child_s::Int64, parent_s::Int64,
        m_pos::Vector{Float64}, b::Int64,
        fs0m::SensitiveFloat{StarPosParams, NumType},
        fs1m::SensitiveFloat{GalaxyPosParams, NumType},
        E_G::SensitiveFloat{CanonicalParams, NumType},
        var_G::SensitiveFloat{CanonicalParams, NumType},
        wcs_jacobian::Array{Float64, 2})

    # Accumulate over PSF components.
    clear!(fs0m)
    for star_mc in star_mcs[:, parent_s]
        accum_star_pos!(star_mc, m_pos, fs0m, wcs_jacobian)
    end

    clear!(fs1m)
    for i = 1:2 # Galaxy types
        for j in 1:[8,6][i] # Galaxy component
            for k = 1:3 # PSF component
                accum_galaxy_pos!(
                  gal_mcs[k, j, i, parent_s], m_pos, fs1m, wcs_jacobian)
            end
        end
    end

    # Add the contributions of this source in this band to
    # E(G) and Var(G).

    # TODO: You will need to:
    # - square a sensitive float
    # - multiply them (they have different params)
    # - multiply by a (special case?)
    # - log a sensitive float
    # - somehow get x / y^2 (for the variance term)

    # In the structures below, 1 = star and 2 = galaxy.
    a = vs[ids.a]
    fsm = (fs0m, fs1m)
    lf = (sb.E_l_a[b, 1].v * fs0m.v, sb.E_l_a[b, 2].v * fs1m.v)
    llff = (sb.E_ll_a[b, 1].v * fs0m.v^2, sb.E_ll_a[b, 2].v * fs1m.v^2)

    E_G_s_v = a[1] * lf[1] + a[2] * lf[2]
    E_G.v += E_G_s_v

    # These formulas for the variance of G use the fact that the
    # variational distributions of each source and band are independent.
    var_G.v -= E_G_s_v^2
    var_G.v += a[1] * llff[1] + a[2] * llff[2]

    # Add the contributions of this source in this band to
    # the derivatives of E(G) and Var(G).

    # a derivatives:
    for i in 1:Ia
        E_G.d[ids.a[i], child_s] += lf[i]
        var_G.d[ids.a[i], child_s] -= 2 * E_G_s_v * lf[i]
        var_G.d[ids.a[i], child_s] += llff[i]
    end

    # Derivatives with respect to the spatial parameters
    for i in 1:Ia # Stars and galaxies
        for p1 in 1:length(shape_standard_alignment[i])
            p0 = shape_standard_alignment[i][p1]
            a_fd = a[i] * fsm[i].d[p1]
            a_El_fd = sb.E_l_a[b, i].v * a_fd
            E_G.d[p0, child_s] += a_El_fd
            var_G.d[p0, child_s] -= 2 * E_G_s_v * a_El_fd
            var_G.d[p0, child_s] += a_fd * sb.E_ll_a[b, i].v * 2 * fsm[i].v
        end
    end

    # Derivatives with respect to the brightness parameters.
    for i in 1:Ia # Stars and galaxies
        # TODO: use p1, once using BrightnessParams type
        for p1 in 1:length(brightness_standard_alignment[i])
            p0 = brightness_standard_alignment[i][p1]
            a_f_Eld = a[i] * fsm[i].v * sb.E_l_a[b, i].d[p0]
            E_G.d[p0, child_s] += a_f_Eld
            var_G.d[p0, child_s] -= 2 * E_G_s_v * a_f_Eld
            var_G.d[p0, child_s] += a[i] * fsm[i].v^2 * sb.E_ll_a[b, i].d[p0]
        end
    end
end


@doc """
Add the contributions of the expected value of a G term to the ELBO.

Args:
  - tile_sources: A vector of source ids influencing this tile
  - x_nbm: The photon count at this pixel
  - iota: The optical sensitivity
  - E_G: The variational expected value of G
  - var_G: The variational variance of G
  - accum: A SensitiveFloat for the ELBO which is updated

Returns:
  - Adds the contributions of E_G and var_G to accum in place.
""" ->
function accum_pixel_ret!{NumType <: Number}(
        tile_sources::Vector{Int64},
        x_nbm::Float64, iota::Float64,
        E_G::SensitiveFloat{CanonicalParams, NumType},
        var_G::SensitiveFloat{CanonicalParams, NumType},
        ret::SensitiveFloat{CanonicalParams, NumType})
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


@doc """
Expected pixel brightness.
Args:
  h: The row of the tile
  w: The column of the tile
  ...the rest are the same as elsewhere.

Returns:
  - Iota.
""" ->
function expected_pixel_brightness!{NumType <: Number}(
    h::Int64, w::Int64,
    sbs::Vector{SourceBrightness{NumType}},
    star_mcs::Array{BvnComponent{NumType}, 2},
    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
    tile::ImageTile,
    E_G::SensitiveFloat{CanonicalParams, NumType},
    var_G::SensitiveFloat{CanonicalParams, NumType},
    mp::ModelParams{NumType},
    tile_sources::Vector{Int64},
    fs0m::SensitiveFloat{StarPosParams, NumType},
    fs1m::SensitiveFloat{GalaxyPosParams, NumType};
    include_epsilon::Bool=true)

  clear!(E_G)
  clear!(var_G)

  if include_epsilon
    E_G.v = tile.constant_background ? tile.epsilon : tile.epsilon_mat[h, w]
  else
    E_G.v = 0.0
  end

  for child_s in 1:length(tile_sources)
      accum_pixel_source_stats!(
          sbs[tile_sources[child_s]], star_mcs, gal_mcs,
          mp.vp[tile_sources[child_s]], child_s, tile_sources[child_s],
          Float64[tile.h_range[h], tile.w_range[w]], tile.b,
          fs0m, fs1m, E_G, var_G,
          mp.patches[child_s, tile.b].wcs_jacobian)
  end

  # Return the appropriate value of iota.
  tile.constant_background ? tile.iota : tile.iota_vec[h]
end


@doc """
A type containing all the information that needs to be communicated
to worker nodes at each iteration.  This currently consists of pre-computed
information about each source.

Attributes:
  vp: The VariationalParams for the ModelParams object
  star_mcs_vec: A vector of star BVN components, one for each band
  gal_mcs_vec: A vector of galaxy BVN components, one for each band
  sbs_vec: A vector of brightness vectors, one for each band
""" ->
type ParameterMessage{NumType <: Number}
  vp::VariationalParams{NumType}
  star_mcs_vec::Vector{Array{BvnComponent{NumType},2}}
  gal_mcs_vec::Vector{Array{GalaxyCacheComponent{NumType},4}}
  sbs_vec::Vector{Vector{SourceBrightness{NumType}}}
end

@doc """
This allocates memory for but does not initialize the source parameters.
""" ->
ParameterMessage{NumType <: Number}(mp::ModelParams{NumType}) = begin
  num_bands = size(mp.patches)[2]
  star_mcs_vec = Array(Array{BvnComponent{NumType},2}, num_bands)
  gal_mcs_vec = Array(Array{GalaxyCacheComponent{NumType},4}, num_bands)
  sbs_vec = Array(Vector{SourceBrightness{NumType}}, num_bands)
  ParameterMessage(mp.vp, star_mcs_vec, gal_mcs_vec, sbs_vec)
end


@doc """
Update a ParameterMessage in place using mp.

Args:
  - mp: A ModelParams object
  - param_msg: A ParameterMessage that is updated using the parameter values
               in mp.
""" ->
function update_parameter_message!{NumType <: Number}(
    mp::ModelParams{NumType}, param_msg::ParameterMessage{NumType})
  for b=1:5
    param_msg.star_mcs_vec[b], param_msg.gal_mcs_vec[b] =
      load_bvn_mixtures(mp, b);
    param_msg.sbs_vec[b] = SourceBrightness{NumType}[
      SourceBrightness(mp.vp[s]) for s in 1:mp.S];
  end
end


@doc """
Add a tile's contribution to the ELBO likelihood term by
modifying accum in place.

Args:
  - tile: An image tile.
  - mp: The current model parameters.
  - sbs: The current source brightnesses.
  - star_mcs: All the star * PCF components.
  - gal_mcs: All the galaxy * PCF components.
  - accum: The ELBO log likelihood to be updated.
""" ->
function tile_likelihood!{NumType <: Number}(
        tile::ImageTile,
        tile_sources::Vector{Int64},
        mp::ModelParams{NumType},
        sbs::Vector{SourceBrightness{NumType}},
        star_mcs::Array{BvnComponent{NumType}, 2},
        gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
        accum::SensitiveFloat{CanonicalParams, NumType};
        include_epsilon::Bool=true)

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
            accum.v += tile_x * log(ep) - num_pixels * ep
        else
            for w in 1:tile.w_width, h in 1:tile.h_width
                this_pixel = tile.pixels[h, w]
                if !Base.isnan(this_pixel)
                    ep = tile.epsilon_mat[h, w]
                    accum.v += this_pixel * log(ep) - ep
                end
            end
        end
        return
    end

    # fs0m and fs1m accumulate contributions from all sources.
    fs0m = zero_sensitive_float(StarPosParams, NumType)
    fs1m = zero_sensitive_float(GalaxyPosParams, NumType)

    tile_S = length(tile_sources)
    E_G = zero_sensitive_float(CanonicalParams, NumType, tile_S)
    var_G = zero_sensitive_float(CanonicalParams, NumType, tile_S)

    # Iterate over pixels that are not NaN.
    for w in 1:tile.w_width, h in 1:tile.h_width
        this_pixel = tile.pixels[h, w]
        if !Base.isnan(this_pixel)
            iota = expected_pixel_brightness!(
              h, w, sbs, star_mcs, gal_mcs, tile, E_G, var_G,
              mp, tile_sources, fs0m, fs1m,
              include_epsilon=include_epsilon)
            accum_pixel_ret!(tile_sources, this_pixel, iota, E_G, var_G, accum)
        end
    end

    # Subtract the log factorial term
    accum.v += -sum(lfact(tile.pixels[!Base.isnan(tile.pixels)]))
end


@doc """
Return the image predicted for the tile given the current parameters.

Args:
  - tile: An image tile.
  - mp: The current model parameters.
  - sbs: The current source brightnesses.
  - star_mcs: All the star * PCF components.
  - gal_mcs: All the galaxy * PCF components.
  - accum: The ELBO log likelihood to be updated.

Returns:
  A matrix of the same size as the tile with the predicted brightnesses.
""" ->
function tile_predicted_image{NumType <: Number}(
        tile::ImageTile,
        tile_sources::Vector{Int64},
        mp::ModelParams{NumType},
        sbs::Vector{SourceBrightness{NumType}},
        star_mcs::Array{BvnComponent{NumType}, 2},
        gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
        accum::SensitiveFloat{CanonicalParams, NumType};
        include_epsilon::Bool=true)

    # fs0m and fs1m accumulate contributions from all sources.
    fs0m = zero_sensitive_float(StarPosParams, NumType)
    fs1m = zero_sensitive_float(GalaxyPosParams, NumType)

    tile_S = length(tile_sources)
    E_G = zero_sensitive_float(CanonicalParams, NumType, tile_S)
    var_G = zero_sensitive_float(CanonicalParams, NumType, tile_S)

    predicted_pixels = copy(tile.pixels)
    # Iterate over pixels that are not NaN.
    for w in 1:tile.w_width, h in 1:tile.h_width
        this_pixel = tile.pixels[h, w]
        if !Base.isnan(this_pixel)
            iota = expected_pixel_brightness!(
              h, w, sbs, star_mcs, gal_mcs, tile, E_G, var_G,
              mp, tile_sources, fs0m, fs1m,
              include_epsilon=include_epsilon)
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
  star_mcs, gal_mcs = load_bvn_mixtures(mp, b)
  sbs = [SourceBrightness(mp.vp[s]) for s in 1:mp.S]

  accum = zero_sensitive_float(CanonicalParams, NumType, mp.S)
  tile_sources = mp.tile_sources[b][tile.hh, tile.ww]

  tile_predicted_image(tile,
                       tile_sources,
                       mp,
                       sbs,
                       star_mcs,
                       gal_mcs,
                       accum,
                       include_epsilon=include_epsilon)
end


@doc """
The ELBO likelihood for given brighntess and bvn components.
""" ->
function elbo_likelihood!{NumType <: Number}(
  tiled_image::Array{ImageTile},
  mp::ModelParams{NumType},
  sbs::Vector{SourceBrightness{NumType}},
  star_mcs::Array{BvnComponent{NumType}, 2},
  gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
  accum::SensitiveFloat{CanonicalParams, NumType})

  @assert maximum(mp.active_sources) <= mp.S
  for tile in tiled_image[:]
    tile_sources = mp.tile_sources[tile.b][tile.hh, tile.ww]
    if length(intersect(tile_sources, mp.active_sources)) > 0
      tile_likelihood!(
        tile, tile_sources, mp, sbs, star_mcs, gal_mcs, accum);
    end
  end

end


@doc """
Evaluate the ELBO with pre-computed brightnesses and components
stored in ParameterMessage.
""" ->
function elbo_likelihood!{NumType <: Number}(
    tiled_blob::TiledBlob,
    param_msg::ParameterMessage{NumType},
    mp::ModelParams{NumType},
    accum::SensitiveFloat{CanonicalParams, NumType})

  clear!(accum)
  mp.vp = param_msg.vp
  for b in 1:5
    sbs = param_msg.sbs_vec[b]
    star_mcs = param_msg.star_mcs_vec[b]
    gal_mcs = param_msg.gal_mcs_vec[b]
    elbo_likelihood!(tiled_blob[b], mp, sbs, star_mcs, gal_mcs, accum)
  end
end


@doc """
Add the expected log likelihood ELBO term for an image to accum.

Args:
  - tiles: An array of ImageTiles
  - mp: The current model parameters.
  - accum: A sensitive float containing the ELBO.
  - b: The current band
""" ->
function elbo_likelihood!{NumType <: Number}(
  tiles::Array{ImageTile}, mp::ModelParams{NumType},
  b::Int64, accum::SensitiveFloat{CanonicalParams, NumType})

  star_mcs, gal_mcs = load_bvn_mixtures(mp, b)
  sbs = SourceBrightness{NumType}[SourceBrightness(mp.vp[s]) for s in 1:mp.S]
  elbo_likelihood!(tiles, mp, sbs, star_mcs, gal_mcs, accum)
end


@doc """
Return the expected log likelihood for all bands in a section
of the sky.
""" ->
function elbo_likelihood{NumType <: Number}(
  tiled_blob::TiledBlob, mp::ModelParams{NumType})
    # Return the expected log likelihood for all bands in a section
    # of the sky.

    ret = zero_sensitive_float(CanonicalParams, NumType, mp.S)
    for b in 1:length(tiled_blob)
        elbo_likelihood!(tiled_blob[b], mp, b, ret)
    end
    ret
end


@doc """
Calculates and returns the ELBO and its derivatives for all the bands
of an image.

Args:
  - tiled_blob: A TiledBlob.
  - mp: Model parameters.
""" ->
function elbo{NumType <: Number}(tiled_blob::TiledBlob, mp::ModelParams{NumType})
    ret = elbo_likelihood(tiled_blob, mp)
    subtract_kl!(mp, ret)
    ret
end



end
