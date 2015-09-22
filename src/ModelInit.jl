# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module ModelInit

VERSION < v"0.4.0-dev" && using Docile

export sample_prior, peak_init

export intialize_celeste, initialize_model_params

using FITSIO
using Distributions
using Util
using CelesteTypes

import SloanDigitalSkySurvey: WCS
import Images
import WCSLIB
import CelesteTypes.SkyPatch


function sample_prior()
    const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

    stars_file = open("$dat_dir/priors/stars.dat")
    r_fit1, k1, cmean1, ccov1 = deserialize(stars_file)
    close(stars_file)

    gals_file = open("$dat_dir/priors/gals.dat")
    r_fit2, k2, cmean2, ccov2 = deserialize(gals_file)
    close(gals_file)

    # TODO: use r_fit1 and r_fit2 instead of magic numbers ?

    # magic numbers below determined from the output of primary
    # on the test set of stamps
    PriorParams(
        [0.28, 0.72],                       # a
        [(0.47, 1. / 0.012), (1.28, 1. / 0.11)], # r
        Vector{Float64}[k1, k2],            # k
        [(cmean1, ccov1), (cmean2, ccov2)]) # c
end

@doc """
Return a default-initialized VariationalParams object.
""" ->
function init_source(init_pos::Vector{Float64})
    #TODO: use blob (and perhaps priors) to initialize these sensibly
    ret = Array(Float64, length(CanonicalParams))
    ret[ids.a[2]] = 0.5
    ret[ids.a[1]] = 1.0 - ret[ids.a[2]]
    ret[ids.u[1]] = init_pos[1]
    ret[ids.u[2]] = init_pos[2]
    ret[ids.r1] = 1e3
    ret[ids.r2] = 2e-3
    ret[ids.e_dev] = 0.5
    ret[ids.e_axis] = 0.5
    ret[ids.e_angle] = 0.
    ret[ids.e_scale] = 1.
    ret[ids.k] = 1. / size(ids.k, 1)
    ret[ids.c1] = 0.
    ret[ids.c2] =  1e-2
    ret
end

@doc """
Return a VariationalParams object initialized form a catalog entry.
""" ->
function init_source(ce::CatalogEntry)
    # TODO: sync this up with the transform bounds
    ret = init_source(ce.pos)

    ret[ids.r1[1]] = max(0.0001, ce.star_fluxes[3]) ./ ret[ids.r2[1]]
    ret[ids.r1[2]] = max(0.0001, ce.gal_fluxes[3]) ./ ret[ids.r2[2]]

    get_color(c2, c1) = begin
        c2 > 0 && c1 > 0 ? min(max(log(c2 / c1), -9.), 9.) :
            c2 > 0 && c1 <= 0 ? 3.0 :
                c2 <= 0 && c1 > 0 ? -3.0 : 0.0
    end
    get_colors(raw_fluxes) = begin
        [get_color(raw_fluxes[c+1], raw_fluxes[c]) for c in 1:4]
    end

    ret[ids.c1[:, 1]] = get_colors(ce.star_fluxes)
    ret[ids.c1[:, 2]] = get_colors(ce.gal_fluxes)

    ret[ids.e_dev] = min(max(ce.gal_frac_dev, 0.015), 0.985)

    ret[ids.e_axis] = ce.is_star ? .8 : min(max(ce.gal_ab, 0.015), 0.985)
    ret[ids.e_angle] = ce.gal_angle
    ret[ids.e_scale] = ce.is_star ? 0.2 : min(max(ce.gal_scale, 0.2), 69.)

    ret
end


function matched_filter(img::Image)
    H, W = 5, 5
    kernel = zeros(Float64, H, W)
    for k in 1:3
        mvn = MvNormal(img.psf[k].xiBar, img.psf[k].tauBar)
        for h in 1:H
            for w in 1:W
                x = [h - (H + 1) / 2., w - (W + 1) / 2.]
                kernel[h, w] += img.psf[k].alphaBar * pdf(mvn, x)
            end
        end
    end
    kernel /= sum(kernel)
end


function convolve_image(img::Image)
    # Not totally sure why this is helpful,
    # but it may help find
    # peaks in an image that has already been gaussian-blurred.
    # (Ref in ICML & NIPS papers).
    kernel = matched_filter(img)
    H, W = size(img.pixels)
    padded_pixels = Array(Float64, H + 8, W + 8)
    fill!(padded_pixels, median(img.pixels))
    padded_pixels[5:H+4,5:W+4] = img.pixels
    conv2(padded_pixels, kernel)[7:H+6, 7:W+6]
end


function peak_starts(blob::Blob)
    # Heuristically find the peaks in the blob.  (Blob == field)
    H, W = size(blob[1].pixels)
    added_pixels = zeros(Float64, H, W)
    for b in 1:5
        added_pixels += convolve_image(blob[b])
    end
    spread = quantile(added_pixels[:], .7) - quantile(added_pixels[:], .2)
    threshold = median(added_pixels) + 3spread

    peaks = Array(Vector{Float64}, 0)
    i = 0
    for h=3:(H-3), w=3:(W-3)
        if added_pixels[h, w] > threshold &&
                added_pixels[h, w] > maximum(added_pixels[h-2:h+2, w-2:w+2]) - .1
            i += 1
#            println("found peak $i: ", h, " ", w)
#            println(added_pixels[h-3:min(h+3,99), w-3:min(w+3,99)])
            push!(peaks, [h, w])
        end
    end

    R = length(peaks)
    peaks_mat = Array(Float64, 2, R)
    for i in 1:R
        peaks_mat[:, i] = peaks[i]
    end

    peaks_mat
#    wcsp2s(img.wcs, peaks_mat)
end


function peak_init(blob::Blob)
    v1 = peak_starts(blob)
    S = size(v1)[2]
    vp = [init_source(v1[:, s]) for s in 1:S]
    twice_radius = float(max(blob[1].H, blob[1].W))
    ModelParams(vp, sample_prior())
end



#############################
# Set patch sizes using the catalog.

function get_psf_width(psf::Array{PsfComponent}; width_scale=1.0)
  # A heuristic measure of the PSF width based on an anology
  # with it being a mixture of normals.  Note that it is not an actual
  # mixture of normals, and in particular that sum(alphaBar) \ne 1.

  # The PSF is not necessarily centered at (0, 0), but we want a measure
  # of its maximal width around (0, 0), not around its center.
  # Approximate this by finding the covariance of a point randomly drawn
  # from a mixture of gaussians.
  alpha_norm = sum([ psf_comp.alphaBar for psf_comp in psf ])
  cov_est = zeros(Float64, 2, 2)
  for psf_comp in psf
    cov_est +=
      psf_comp.alphaBar * (psf_comp.xiBar * psf_comp.xiBar' + psf_comp.tauBar) /
      alpha_norm
  end

  # Return the twice the sd of the most spread direction, scaled by the total
  # mass in the PSF.
  width_scale * sqrt(eigs(cov_est; nev=1)[1][1]) * alpha_norm
end


@doc """
Choose a reasonable patch radius based on the catalog.
""" ->
function choose_patch_radius(
  pixel_center::Vector{Float64}, ce::CatalogEntry,
  psf::Array{PsfComponent}, img::Image; width_scale=1.0)

    psf_width = get_psf_width(psf, width_scale=width_scale)

    # The galaxy scale is the point with half the light -- if the light
    # were entirely in a univariate normal, this would be at 0.67 standard
    # deviations.  We are being a bit conservative here.
    obj_width =
      ce.is_star ? psf_width: width_scale * ce.gal_scale / 0.67 + psf_width

    if img.constant_background
      epsilon = img.epsilon
    else
      # Get the average sky noise in a rectangle of the width of the psf.
      h_range = (pixel_center[1] - obj_width):(pixel_center[1] + obj_width)
      w_range = (pixel_center[2] - obj_width):(pixel_center[2] + obj_width)
      epsilon = mean(img.epsilon_mat[h_range, w_range])
    end
    flux = ce.is_star ? ce.star_fluxes[img.b] : ce.gal_fluxes[img.b]

    # For the worst-case scenario, find where a 1d pdf would take the
    # value of 5% of the sky noise when the standard deviation is obj_width.
    pdf_target = epsilon / (20 * flux)  # 5% of sky
    rhs = log(pdf_target) + 0.5 * log(2pi) + log(obj_width)
    radius_req = sqrt(-2 * (obj_width ^ 2) * rhs)
    radius_req
end


@doc """
Initialize a SkyPatch object at a particular location.

Args:
  - world_center: The location of the patch
  - radius: The radius, in world coordinates, of the circle of pixels that
            affect this patch of sky.
  - img: An Image object.

Returns:
  A SkyPatch object.
""" ->
SkyPatch(world_center::Vector{Float64},
         radius::Float64, img::Image; fit_psf=true) = begin
    if fit_psf
      psf = Images.get_source_psf(world_center, img)
    else
      psf = img.psf
    end

    pixel_center = WCS.world_to_pixel(img.wcs, world_center)
    wcs_jacobian = WCS.pixel_world_jacobian(img.wcs, pixel_center)

    SkyPatch(world_center, radius, psf, wcs_jacobian, pixel_center)
end


@doc """
Initialize a SkyPatch object for a catalog entry.

Args:
  - ce: The catalog entry for the object.
  - img: An Image object.
  - fit_psf: Whether to fit the psf at this location.

Returns:
  A SkyPatch object with a radius chosen based on the catalog.
""" ->
SkyPatch(ce::CatalogEntry, img::Image; fit_psf=true) = begin
    world_center = ce.pos
    if fit_psf
      psf = Images.get_source_psf(world_center, img)
    else
      psf = img.psf
    end

    pixel_center = WCS.world_to_pixel(img.wcs, world_center)
    wcs_jacobian = WCS.pixel_world_jacobian(img.wcs, pixel_center)

    pix_radius = choose_patch_radius(pixel_center, ce, psf, img)
    sky_radius = Images.pixel_radius_to_world(pix_radius, wcs_jacobian)

    SkyPatch(world_center, sky_radius, psf, wcs_jacobian, pixel_center)
end


@doc """
Get the sources associated with each tile in a TiledImage.

Args:
  - tiled_image: A TiledImage
  - wcs: The world coordinate system for the object.
  - patches: A vector of SkyPatch objects, one for each celestial object.

Returns:
  - An array (same dimensions as the tiles) of vectors of indices
    into patches indicating which patches are affected by any pixels
    in the tiles.
""" ->
function get_tiled_image_sources(
  tiled_image::TiledImage, wcs::WCSLIB.wcsprm, patches::Vector{SkyPatch})

  H, W = size(tiled_image)
  tile_sources = fill(Int64[], H, W)
  candidates = Images.local_source_candidates(tiled_image, patches)
  for h in 1:H, w in 1:W
    # Only look for sources within the candidate set.
    cand_patches = patches[candidates[h, w]]
    if length(cand_patches) > 0
      cand_sources = Images.local_sources(tiled_image[h, w], cand_patches, wcs)
      tile_sources[h, w] = candidates[h, w][cand_sources]
    else
      tile_sources[h, w] = Int64[]
    end
  end
  tile_sources
end


# @doc """
# Break the images into tiles and initialize the model parameters.
#
# Args:
#   - blob: A Blob containing the images.
#   - mp: Model parameters.
#
# Returns:
#   Updates in place mp's patches and tile sources.
#   Returns a tiled blob.
# """ ->
# function initialize_tiles_and_patches!(
#     blob::Blob, mp::ModelParams, tile_width::Int64;
#     patch_radius=Inf, fit_psf=true)
#   tiled_blob = Images.break_blob_into_tiles(blob, tile_width)
#   initialize_tiles_and_patches!(
#     tiled_blob, blob, mp, patch_radius=patch_radius, fit_psf=fit_psf)
#   tiled_blob
# end
#
#
# @doc """
# Initialize the tiles and patches if you've already tiled your blob
# (e.g. when you are cropping to a single object location)
# """ ->
# function initialize_tiles_and_patches!(
#     tiled_blob::TiledBlob, blob::Blob, mp::ModelParams;
#     patch_radius=Inf, fit_psf=true)
#   # Set the model parameters
#
#   @assert length(tiled_blob) == length(blob)
#   mp.patches = Array(SkyPatch, mp.S, length(blob))
#   mp.tile_sources = Array(Array{Array{Int64}}, length(blob))
#
#   for b = 1:length(blob)
#     for s=1:mp.S
#       mp.patches[s, b] =
#         SkyPatch(mp.vp[s][ids.u], patch_radius, blob[b], fit_psf=fit_psf)
#     end
#     mp.tile_sources[b] =
#       get_tiled_image_sources(tiled_blob[b], blob[b].wcs, mp.patches[:, b][:])
#   end
#
#   tiled_blob
# end


@doc """
Turn a blob and vector of catalog entries into a tiled_blob and model
parameters that can be used with Celeste.
""" ->
function initialize_celeste(
    blob::Blob, cat::Vector{CatalogEntry};
    tile_width::Int64=typemax(Int64), fit_psf::Bool=true,
    patch_radius::Float64=-1., radius_from_cat::Bool=true)

  tiled_blob = Images.break_blob_into_tiles(blob, tile_width)
  mp = initialize_model_params(tiled_blob, blob, cat,
                               fit_psf=fit_psf, patch_radius=patch_radius,
                               radius_from_cat=radius_from_cat)
  tiled_blob, mp
end


@doc """
Initilize the model params to the given catalog and tiled image.

Args:
  - tiled_blob: A TiledBlob
  - blob: The original Blob
  - cat: A vector of catalog entries
  - fit_psf: Whether to give each patch its own local PSF
  - patch_radius: If set less than Inf or if radius_from_cat=false,
                  the radius in world coordinates of each patch.
  - radius_from_cat: If true, choose the patch radius from the catalog.
                     If false, use patch_radius for each patch.
""" ->
function initialize_model_params(
    tiled_blob::TiledBlob, blob::Blob, cat::Vector{CatalogEntry};
    fit_psf::Bool=true, patch_radius::Float64=-1., radius_from_cat::Bool=true)

  @assert length(tiled_blob) == length(blob)

  # If patch_radius is set by the caller, don't use the radius from the catalog.
  if patch_radius != -1.
    radius_from_cat = false
  end
  if !radius_from_cat
    @assert(patch_radius > 0.,
            "If !radius_from_cat, you must specify a positive patch_radius.")
  end
  vp = [init_source(ce) for ce in cat]
  mp = ModelParams(vp, sample_prior())

  mp.patches = Array(SkyPatch, mp.S, length(blob))
  mp.tile_sources = Array(Array{Array{Int64}}, length(blob))
  for b = 1:length(blob)
    for s=1:mp.S
      if radius_from_cat
        mp.patches[s, b] = SkyPatch(ce[s], blob[b], fit_psf=fit_psf)
      else
        mp.patches[s, b] =
          SkyPatch(mp.vp[s][ids.u], patch_radius, blob[b], fit_psf=fit_psf)
      end
    end
    mp.tile_sources[b] =
      get_tiled_image_sources(tiled_blob[b], blob[b].wcs, mp.patches[:, b][:])
  end

end

end
