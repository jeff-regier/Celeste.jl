module ModelInit

export sample_prior,
       peak_init,
       intialize_celeste,
       initialize_model_params,
       get_relevant_sources

using FITSIO
using Distributions
import Logging

import ..WCSUtils
import WCS.WCSTransform

using ..Types
import ..PSF
import ..ElboDeriv  # for trim_source_tiles
import ..SkyImages
import ..Types: SkyPatch


const cfgdir = joinpath(Pkg.dir("Celeste"), "cfg")

function sample_prior()

    # set a = [.99, .01] if stars are underrepresented
    # due to the greater flexibility of the galaxy model
    #a = [0.28, 0.72]
    a = [0.99, 0.01]
    r_mean = Array(Float64, Ia)
    r_var = Array(Float64, Ia)
    k = Array(Float64, D, Ia)
    c_mean = Array(Float64, B - 1, D, Ia)
    c_cov = Array(Float64, B - 1, B - 1, D, Ia)

    stars_file = open(joinpath(cfgdir, "stars$D.dat"))
    r_fit1, k[:, 1], c_mean[:,:,1], c_cov[:,:,:,1] = deserialize(stars_file)
    close(stars_file)

    gals_file = open(joinpath(cfgdir, "gals$D.dat"))
    r_fit2, k[:, 2], c_mean[:,:,2], c_cov[:,:,:,2] = deserialize(gals_file)
    close(gals_file)

    # These "magic numbers" have been in use for a while.
    # They were initially gamma parameters, and now they are log normal
    # parameters.  TODO: Get rid of these and use an empirical prior.
    # r = [0.47 1.28; 1/0.012 1/0.11] # These were gamma (shape, scale)

    mean_brightness = [0.47 / 0.012, 1.28 / 0.11 ]
    var_brightness = [0.47 / (0.012 ^ 2), 1.28 / (0.11 ^ 2) ]

    # The prior contains parameters of a lognormal distribution with
    # the desired means.
    r_var = log(var_brightness ./ (mean_brightness .^ 2) + 1)
    r_mean = log(mean_brightness) - 0.5 * r_var
    PriorParams(a, r_mean, r_var, k, c_mean, c_cov)
end


"""
Return a default-initialized VariationalParams object.
"""
function init_source(init_pos::Vector{Float64})
    #TODO: use blob (and perhaps priors) to initialize these sensibly
    ret = Array(Float64, length(CanonicalParams))
    ret[ids.a[2]] = 0.5
    ret[ids.a[1]] = 1.0 - ret[ids.a[2]]
    ret[ids.u[1]] = init_pos[1]
    ret[ids.u[2]] = init_pos[2]
    ret[ids.r1] = log(2.0)
    ret[ids.r2] = 1e-3
    ret[ids.e_dev] = 0.5
    ret[ids.e_axis] = 0.5
    ret[ids.e_angle] = 0.
    ret[ids.e_scale] = 1.
    ret[ids.k] = 1. / size(ids.k, 1)
    ret[ids.c1] = 0.
    ret[ids.c2] =  1e-2
    ret
end

"""
Return a VariationalParams object initialized form a catalog entry.
"""
function init_source(ce::CatalogEntry)
    # TODO: sync this up with the transform bounds
    ret = init_source(ce.pos)

    ret[ids.r1[1]] = log(max(0.1, ce.star_fluxes[3]))
    ret[ids.r1[2]] = log(max(0.1, ce.gal_fluxes[3]))

    function get_color(c2, c1)
        c2 > 0 && c1 > 0 ? min(max(log(c2 / c1), -9.), 9.) :
            c2 > 0 && c1 <= 0 ? 3.0 :
                c2 <= 0 && c1 > 0 ? -3.0 : 0.0
    end

    function get_colors(raw_fluxes)
        [get_color(raw_fluxes[c+1], raw_fluxes[c]) for c in 1:4]
    end

    ret[ids.c1[:, 1]] = get_colors(ce.star_fluxes)
    ret[ids.c1[:, 2]] = get_colors(ce.gal_fluxes)

    ret[ids.e_dev] = min(max(ce.gal_frac_dev, 0.015), 0.985)

    ret[ids.e_axis] = ce.is_star ? .8 : min(max(ce.gal_ab, 0.015), 0.985)
    ret[ids.e_angle] = ce.gal_angle
    ret[ids.e_scale] = ce.is_star ? 0.2 : max(ce.gal_scale, 0.2)

    ret
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


"""
Choose a reasonable patch radius based on the catalog.
TODO: Select this by rendering the object and solving an optimization
problem.

Args:
  - pixel_center: The pixel location of the object
  - ce: The catalog entry for the object
  - psf: The psf at this location
  - img: The image
  - width_scale: A multiple of standard deviations to use
  - max_radius: The maximum radius in pixels.

Returns:
  - A radius in pixels chosen from the catalog entry.
"""
function choose_patch_radius(
  pixel_center::Vector{Float64}, ce::CatalogEntry,
  psf::Array{PsfComponent}, img::Image; width_scale=1.0, max_radius=100)

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
      h_max, w_max = size(img.epsilon_mat)
      h_lim = [Int(floor((pixel_center[1] - obj_width))),
                       Int(ceil((pixel_center[1] + obj_width)))]
      w_lim = [Int(floor((pixel_center[2] - obj_width))),
                       Int(ceil((pixel_center[2] + obj_width)))]
      h_range = max(h_lim[1], 1):min(h_lim[2], h_max)
      w_range = max(w_lim[1], 1):min(w_lim[2], w_max)
      epsilon = mean(img.epsilon_mat[h_range, w_range])
    end
    flux = ce.is_star ? ce.star_fluxes[img.b] : ce.gal_fluxes[img.b]
    @assert flux > 0.

    # Choose enough pixels that the light is either 90% of the light
    # would be captured from a 1d gaussian or 5% of the sky noise,
    # whichever is a larger radius.
    pdf_90 = exp(-0.5 * (1.64)^2) / (sqrt(2pi) * obj_width)
    pdf_target = min(pdf_90, epsilon / (20 * flux))
    rhs = log(pdf_target) + 0.5 * log(2pi) + log(obj_width)
    radius_req = sqrt(-2 * (obj_width ^ 2) * rhs)
    min(radius_req, max_radius)
end


"""
Initialize a SkyPatch object at a particular location.

Args:
  - world_center: The location of the patch
  - world_radius: The radius, in world coordinates, of the circle of pixels
                  that affect this patch of sky.
  - img: An Image object.

Returns:
  A SkyPatch object.

Note: this is only used in tests.
"""
function SkyPatch(world_center::Vector{Float64}, world_radius::Float64,
                  img::Image; fit_psf=true)
    psf = fit_psf ? PSF.get_source_psf(world_center, img)[1] : img.psf
    pixel_center = WCSUtils.world_to_pix(img.wcs, world_center)
    wcs_jacobian = WCSUtils.pixel_world_jacobian(img.wcs, pixel_center)
    radius_pix = maxabs(eigvals(wcs_jacobian)) * world_radius

    SkyPatch(world_center, radius_pix, psf, wcs_jacobian, pixel_center)
end


"""
Initialize a SkyPatch object for a catalog entry.

Args:
  - ce: The catalog entry for the object.
  - img: An Image object.
  - fit_psf: Whether to fit the psf at this location.
  - scale_patch_size: A hack.  Scale the catalog-chosen patch size.

Returns:
  A SkyPatch object with a radius chosen based on the catalog.
"""
function SkyPatch(ce::CatalogEntry, img::Image; fit_psf=true,
                  scale_patch_size=1.0)
    world_center = ce.pos
    psf = fit_psf ? PSF.get_source_psf(world_center, img)[1] : img.psf
    pixel_center = WCSUtils.world_to_pix(img.wcs, world_center)
    wcs_jacobian = WCSUtils.pixel_world_jacobian(img.wcs, pixel_center)
    radius_pix = choose_patch_radius(pixel_center, ce, psf, img)

    SkyPatch(world_center, scale_patch_size * radius_pix, psf,
             wcs_jacobian, pixel_center)
end

"""
Initialize a SkyPatch from an existing SkyPatch and a new PSF.
"""
SkyPatch(patch::SkyPatch, psf::Vector{PsfComponent}) =
    SkyPatch(patch.center, patch.radius_pix, psf, patch.wcs_jacobian,
             patch.pixel_center)


"""
Update ModelParams with the PSFs for a range of object ids.

Args:
  - mp: A ModelParams whose patches will be updated.
  - relevant_sources: A vector of source ids that index into mp.patches
  - blob: A vector of images.

Returns:
  - Updates mp.patches in place with fitted psfs for each source in
    relevant_sources.
"""
function fit_object_psfs!{NumType <: Number}(
    mp::ModelParams{NumType}, target_sources::Vector{Int}, blob::Blob)

  # Initialize an optimizer
  initial_psf_params = PSF.initialize_psf_params(psf_K, for_test=false);
  psf_transform = PSF.get_psf_transform(initial_psf_params);
  psf_optimizer = PSF.PsfOptimizer(psf_transform, psf_K);

  @assert size(mp.patches, 2) == length(blob)

  for b in 1:length(blob)  # loop over images
    Logging.debug("Fitting PSFS for band $b")
    # Get a starting point in the middle of the image.
    pixel_loc = Float64[ blob[b].H / 2.0, blob[b].W / 2.0 ]
    raw_central_psf = blob[b].raw_psf_comp(pixel_loc[1], pixel_loc[2])
    central_psf, central_psf_params =
      PSF.fit_raw_psf_for_celeste(raw_central_psf, psf_optimizer, initial_psf_params)

    # Get all relevant sources *in this image*
    relevant_sources = get_all_relevant_sources_in_image(mp, target_sources, b)

    for s in relevant_sources
      Logging.debug("Fitting PSF for b=$b, source=$s, objid=$(mp.objids[s])")
      patch = mp.patches[s, b]
      # Set the starting point at the center's PSF.
      psf, psf_params =
        PSF.get_source_psf(
          patch.center, blob[b], psf_optimizer, central_psf_params)
      mp.patches[s, b] = ModelInit.SkyPatch(patch, psf)
    end
  end
end

##########################
# Local sources

"""
A fast function to determine which sources might belong to which tiles.

Args:
  - tiles: A TiledImage
  - patch_ctrs: Vector of length-2 vectors, giving pixel center of each patch.
  - patch_radii_px: Radius of each patch.

Returns:
  - An array (over tiles) of a vector of candidate
    source patches.  If a patch is a candidate, it may be within the patch
    radius of a point in the tile, though it might not.
"""
function local_source_candidates(tiles::TiledImage,
                                 patch_ctrs::Vector{Vector{Float64}},
                                 patch_radii_px::Vector{Float64})
    @assert length(patch_ctrs) == length(patch_radii_px)

    candidates = similar(tiles, Vector{Int})

    for h=1:size(tiles, 1), w=1:size(tiles, 2)
        # Find the patches that are less than the radius plus diagonal from the
        # center of the tile.  These are candidates for having some
        # overlap with the tile.
        tile = tiles[h, w]
        tile_center = (mean(tile.h_range), mean(tile.w_range))
        tile_diag = (0.5 ^ 2) * (tile.h_width ^ 2 + tile.w_width ^ 2)

        patch_distances = zeros(length(patch_ctrs))
        for s in 1:length(patch_ctrs)
            patch_distances[s] += (tile_center[1] - patch_ctrs[s][1])^2
            patch_distances[s] += (tile_center[2] - patch_ctrs[s][2])^2
        end
        candidates[h, w] =
            find(patch_distances .<= (tile_diag .+ patch_radii_px) .^ 2)
    end

    return candidates
end


"""
Args:
  - tile: An ImageTile (containing tile coordinates)
  - patch_ctrs: Vector of length-2 vectors, giving pixel center of each patch.
  - patch_radii_px: Radius of each patch.

Returns:
  - A vector of source ids (from 1 to length(patches)) that influence
    pixels in the tile.  A patch influences a tile if
    there is any overlap in their squares of influence.
"""
function get_local_sources(tile::ImageTile,
                           patch_ctrs::Vector{Vector{Float64}},
                           patch_radii_px::Vector{Float64})
    @assert length(patch_ctrs) == length(patch_radii_px)
    tile_sources = Int[]
    tile_ctr = (mean(tile.h_range), mean(tile.w_range))

    for i in eachindex(patch_ctrs)
        patch_ctr = patch_ctrs[i]
        patch_r = patch_radii_px[i]

        # This is a "ball" in the infinity norm.
        if ((abs(tile_ctr[1] - patch_ctr[1]) < patch_r + 0.5 * tile.h_width) &&
            (abs(tile_ctr[2] - patch_ctr[2]) < patch_r + 0.5 * tile.w_width))
            push!(tile_sources, i)
        end


    end

    tile_sources
end


"""
Get the sources associated with each tile in a TiledImage.

Args:
  - tiled_image: A TiledImage
  - patch_ctrs: Vector of length-2 vectors, giving pixel center of each patch.
  - patch_radii_px: Radius of each patch.
Returns:
  - An array (same dimensions as the tiles) of vectors of indices
    into patches indicating which patches are affected by any pixels
    in the tiles.
"""
function get_tiled_image_sources(tiled_image::TiledImage,
                                 patch_ctrs::Vector{Vector{Float64}},
                                 patch_radii_px::Vector{Float64})
    candidates = local_source_candidates(tiled_image, patch_ctrs,
                                         patch_radii_px)

    out = similar(tiled_image, Vector{Int})
    for hh in 1:size(tiled_image, 1), ww in 1:size(tiled_image, 2)
        cands = candidates[hh, ww]  # patches indicies that might overlap tile

        # get indicies in cands that truly overlap the tile.
        idx = get_local_sources(tiled_image[hh, ww],
                                patch_ctrs[cands],
                                patch_radii_px[cands])
        out[hh, ww] = cands[idx]
    end

    return out
end


"""
Turn a blob and vector of catalog entries into a tiled_blob and model
parameters that can be used with Celeste.
"""
function initialize_celeste(
    blob::Blob, cat::Vector{CatalogEntry};
    tile_width::Int=20, fit_psf::Bool=true,
    patch_radius::Float64=-1., radius_from_cat::Bool=true)

  tiled_blob = SkyImages.break_blob_into_tiles(blob, tile_width)
  mp = initialize_model_params(tiled_blob, blob, cat,
                               fit_psf=fit_psf, patch_radius=patch_radius,
                               radius_from_cat=radius_from_cat)
  tiled_blob, mp
end

"""Centers of patches in pixel coordinates"""
patch_ctrs_pix(patches::Vector{SkyPatch}) = [p.pixel_center for p in patches]

"""Radii of patches in pixel coordinates"""
patch_radii_pix(patches::Vector{SkyPatch}) = [p.radius_pix for p in patches]

"""
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
"""
function initialize_model_params(
    tiled_blob::TiledBlob, blob::Blob, cat::Vector{CatalogEntry};
    fit_psf::Bool=true, patch_radius::Float64=-1., radius_from_cat::Bool=true,
    scale_patch_size::Float64=1.0)

    @assert length(tiled_blob) == length(blob)
    @assert(length(cat) > 0,
            "Cannot initilize model parameters with no catalog entries")

    # If patch_radius is set by the caller, don't use the radius from
    # the catalog.
    if patch_radius != -1.
        radius_from_cat = false
    end
    if !radius_from_cat
        @assert(patch_radius > 0.,
                "If !radius_from_cat, you must specify a positive patch_radius.")
    end

    info("Loading variational parameters from catalogs.")
    vp = Array{Float64, 1}[init_source(ce) for ce in cat]
    mp = ModelParams(vp, sample_prior())
    mp.objids = ASCIIString[ cat_entry.objid for cat_entry in cat]

    mp.patches = Array(SkyPatch, mp.S, length(blob))
    mp.tile_sources = Array(Array{Vector{Int}, 2}, length(blob))

    for b = 1:length(blob)
        for s=1:mp.S
            mp.patches[s, b] = radius_from_cat ?
            SkyPatch(cat[s], blob[b], fit_psf=fit_psf,
                     scale_patch_size=scale_patch_size):
            SkyPatch(mp.vp[s][ids.u], patch_radius, blob[b], fit_psf=fit_psf)
        end
        patches = vec(mp.patches[:, b])
        mp.tile_sources[b] = get_tiled_image_sources(tiled_blob[b],
                                                     patch_ctrs_pix(patches),
                                                     patch_radii_pix(patches))
    end
    print("\n")

    # improved initializations
    for i=1:length(mp.vp)
        mp.vp[i][ids.a[1]] = cat[i].is_star ? 0.8: 0.2
        mp.vp[i][ids.a[2]] = 1.0 - mp.vp[i][ids.a][1]
    end

    return mp
end


"""
Return an array of source indices that have some overlap with target_s.

Args:
  - mp: The ModelParams
  - target_s: The index of the source of interest

Returns:
  - An array of integers that index into mp.s representing all sources that
    co-occur in at least one tile with target_s, including target_s itself.
"""
function get_relevant_sources{NumType <: Number}(
    mp::ModelParams{NumType}, target_s::Int)

  relevant_sources = Int[]
  for b = 1:length(mp.tile_sources), tile_sources in mp.tile_sources[b]
    if target_s in tile_sources
      relevant_sources = union(relevant_sources, tile_sources);
    end
  end

  relevant_sources
end


"""
Union of source indicies that have some overlap with any of the input indicies.

Args:

- `mp`: ModelParams
- `idx`: Vector of target source indicies

Returns:

- Array of integers that index into mp.s representing all sources that
  co-occur in at least one tile with *any* of the sources in `idx`.
"""
function get_all_relevant_sources{NumType <: Number}(
    mp::ModelParams{NumType}, idx::Vector{Int})
    out = Int[]
    for s in idx
        out = union(out, get_relevant_sources(mp, s))
    end
    return out
end

"""
    get_all_relevant_sources_in_image(mp, b, idx)

Return indicies of all sources relevant to any of a set of target sources
in the given image.

# Arguments

* `mp::ModelParams`: Model parameters.
* `targets::Vector{Int}`: Indicies of target sources.
* `b::Int`: Index of image.

# Returns

* `Vector{Int}`: Array of integers that index into mp.s. These represent
  all sources that co-occur in at least one tile with *any* of the sources
  in `targets`.
"""
function get_all_relevant_sources_in_image{NumType <: Number}(
    mp::ModelParams{NumType}, target_sources::Vector{Int}, b::Int)

    out = Int[]
    for tile_sources in mp.tile_sources[b]  # loop over image tiles
        # check if *any* of this tile's sources are a target, and
        # if so, add *all* the tile sources to the output.
        if length(intersect(target_sources, tile_sources)) > 0
            out = union(out, tile_sources)
        end
    end
    out
end

"""
Return a reduced Celeste dataset useful for a single object.

Args:
  - objid: An object id in mp_original.objids that you want to fit.
  - mp_original: The original model params with all objects.
  - tiled_blob: The original tiled blob with all the image.
  - blob: The original blob with all the image.
  - cat_entries: The original catalog entries with all the sources.

Returns:
  - trimmed_mp: A ModelParams object containing only the objid source and
      all the sources that co-occur with it.  Its active_sources will be set
      to the objid object.
  - trimmed_tiled_blob: A new TiledBlob with the tiled arrays shrunk to only
      include those necessary for the objid source.

Note that the resulting dataset is only good for fitting the objid source.
The ModelParams object will contain other sources that overlap with the
objid source, but the trimmed_tiled_blob may be missing tiles in which these
overlapping sources occur.

TODO: test!
"""
function limit_to_object_data(
    objid::ASCIIString, mp_original::ModelParams,
    tiled_blob::TiledBlob, blob::Blob, cat_entries::Vector{CatalogEntry})

  @assert length(tiled_blob) == length(blob)
  mp = deepcopy(mp_original)

  s_original = findfirst(mp_original.objids .== objid)
  @assert(s_original > 0, "objid $objid not found in mp_original.")
  mp_original.active_sources = [ s_original ]

  # Get the sources that overlap with this object.
  relevant_sources = get_relevant_sources(mp, s_original)

  trimmed_mp = ModelInit.initialize_model_params(
    tiled_blob, blob, cat_entries[relevant_sources], fit_psf=true);

  s = findfirst(trimmed_mp.objids .== objid)
  trimmed_mp.active_sources = [ s ]

  # Trim to a smaller tiled blob.
  trimmed_tiled_blob = Array(Array{ImageTile}, 5);
  original_tiled_sources = deepcopy(trimmed_mp.tile_sources);
  for b=1:length(tiled_blob)
    hh_vec, ww_vec = ind2sub(size(original_tiled_sources[b]),
      find([ s in sources for sources in original_tiled_sources[b]]))

    hh_range = minimum(hh_vec):maximum(hh_vec);
    ww_range = minimum(ww_vec):maximum(ww_vec);
    trimmed_tiled_blob[b] = tiled_blob[b][hh_range, ww_range];
    trimmed_mp.tile_sources[b] =
      deepcopy(original_tiled_sources[b][hh_range, ww_range]);
  end
  trimmed_tiled_blob = convert(TiledBlob, trimmed_tiled_blob);

  trimmed_mp, trimmed_tiled_blob
end



"""
Set any pixels significantly below background noise for the
specified source to NaN.

Arguments:
  s: The source index that we are trimming to
  mp: The ModelParams object
  tiled_blob: The original tiled blob
  noise_fraction: The proportion of the noise below which we will remove pixels.
  min_radius_pix: A minimum pixel radius to be included.

Returns:
  A new TiledBlob.  Tiles that do not contain the source will be pseudo-tiles
  with empty pixel and noise arrays.  Tiles that contain the source will
  be the same as the original tiles but with NaN where the expected source
  electron counts are below <noise_fraction> of the noise at that pixel.
"""
function trim_source_tiles(
    s::Int, mp::ModelParams{Float64}, tiled_blob::TiledBlob;
    noise_fraction::Float64=0.1, min_radius_pix::Float64=8.0)

  trimmed_tiled_blob =
    Array{ImageTile, 2}[ Array(ImageTile, size(tiled_blob[b])...) for
                         b=1:length(tiled_blob)];

  min_radius_pix_sq = min_radius_pix ^ 2
  for b = 1:length(tiled_blob)
    info("Processing band $b...")

    pix_loc = WCSUtils.world_to_pix(mp.patches[s, b], mp.vp[s][ids.u]);

    H, W = size(tiled_blob[b])
    @assert size(mp.tile_sources[b]) == size(tiled_blob[b])
    for hh=1:H, ww=1:W
      tile = tiled_blob[b][hh, ww];
      tile_sources = mp.tile_sources[b][hh, ww]
      has_source = s in tile_sources
      bright_pixels = Bool[];
      if has_source
        pred_tile_pixels =
          ElboDeriv.tile_predicted_image(tile, mp, [ s ],
                                         include_epsilon=false);
        tile_copy = deepcopy(tiled_blob[b][hh, ww]);

        for h in tile.h_range, w in tile.w_range
          # The pixel location in the rendered image.
          h_im = h - minimum(tile.h_range) + 1
          w_im = w - minimum(tile.w_range) + 1

          keep_pixel = false
          bright_pixel = tile.constant_background ?
            pred_tile_pixels[h_im, w_im] >
              tile.iota * tile.epsilon * noise_fraction:
            pred_tile_pixels[h_im, w_im] >
              tile.iota_vec[h_im] * tile.epsilon_mat[h_im, w_im] * noise_fraction
          close_pixel =
            (h - pix_loc[1]) ^ 2 + (w - pix_loc[2]) ^ 2 < min_radius_pix_sq

          if !(bright_pixel || close_pixel)
            tile_copy.pixels[h_im, w_im] = NaN
          end
        end

        trimmed_tiled_blob[b][hh, ww] = tile_copy;
      else
        # This tile does not contain the source.  Replace the tile with a
        # pseudo-tile that does not have any data in it.
        # The problem is with mp.tile_sources, which can't be allowed to
        # say that an empty tile has a source.
        # TODO: Make a TiledBlob simply an array of an array of tiles
        # rather than a 2d array to avoid this hack.
        empty_tile = ImageTile(b, tile.h_range, tile.w_range,
                               tile.h_width, tile.w_width,
                               Array(Float64, 0, 0), tile.constant_background,
                               tile.epsilon, Array(Float64, 0, 0), tile.iota,
                               Array(Float64, 0))

        trimmed_tiled_blob[b][hh, ww] = empty_tile;
      end
    end
  end
  info("Done trimming.")

  trimmed_tiled_blob
end


end # module
