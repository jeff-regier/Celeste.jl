module ModelInit

import Distributions
import Logging
import WCS

using ..Types
import ..WCSUtils
import ..PSF

const cfgdir = joinpath(Pkg.dir("Celeste"), "cfg")


#######################################
# Functions for matching sources to tiles.

# A pixel circle maps locally to a world ellipse.  Return the major
# axis of that ellipse.
function pixel_radius_to_world(pix_radius::Float64,
                               wcs_jacobian::Matrix{Float64})
  pix_radius / minimum(abs(eigvals(wcs_jacobian)));
end


import WCS.world_to_pix
world_to_pix{T <: Number}(patch::SkyPatch, world_loc::Vector{T}) =
    world_to_pix(patch.wcs_jacobian, patch.center, patch.pixel_center,
                 world_loc)


"""
Return a vector of (h, w) indices of tiles that contain this source.
"""
function find_source_tiles(s::Int, b::Int, mp::ModelParams)
    [ind2sub(size(mp.tile_sources[b]), ind) for ind in
        find([ s in sources for sources in mp.tile_sources[b]])]
end


#############################
# Set patch sizes using the catalog.

function get_psf_width(psf::Array{PsfComponent}; width_scale=1.0)
    # A heuristic measure of the PSF width based on an anology
    # with it being a mixture of normals.    Note that it is not an actual
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
            pixel_center::Vector{Float64},
            ce::CatalogEntry,
            psf::Array{PsfComponent},
            img::Image;
            width_scale=1.0,
            max_radius=100)

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

    for b in 1:length(blob)    # loop over images
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
            mp.patches[s, b] = SkyPatch(patch, psf)
        end
    end
end


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
            tiled_blob::TiledBlob,
            blob::Blob,
            cat::Vector{CatalogEntry};
            fit_psf::Bool=true,
            patch_radius::Float64=NaN)

    @assert length(tiled_blob) == length(blob)
    @assert length(cat) > 0

    Logging.info("Loading variational parameters from catalogs.")

    vp = Array{Float64, 1}[Types.init_source(ce) for ce in cat]
    mp = ModelParams(vp, Types.load_prior())
    mp.objids = ASCIIString[cat_entry.objid for cat_entry in cat]

    mp.patches = Array(SkyPatch, mp.S, length(blob))
    mp.tile_sources = Array(Array{Vector{Int}, 2}, length(blob))

    for b = 1:length(blob)
        img = blob[b]

        for s=1:mp.S
            world_center = cat[s].pos

            pixel_center = WCSUtils.world_to_pix(img.wcs, world_center)
            wcs_jacobian = WCSUtils.pixel_world_jacobian(img.wcs, pixel_center)
            psf = fit_psf ? PSF.get_source_psf(world_center, img)[1] : img.psf

            radius_pix = choose_patch_radius(pixel_center, cat[s], psf, img)

            # for testing
            if !isnan(patch_radius)
                radius_pix = maxabs(eigvals(wcs_jacobian)) * patch_radius
            end

            mp.patches[s, b] = SkyPatch(world_center,
                                        radius_pix,
                                        psf,
                                        wcs_jacobian,
                                        pixel_center)
        end

        mp.tile_sources[b] = Types.get_tiled_image_sources(tiled_blob[b],
                                                           mp.patches[:, b])
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
function get_relevant_sources{NumType <: Number}(mp::ModelParams{NumType},
                                                 target_s::Int)
    relevant_sources = Int[]
    for b = 1:length(mp.tile_sources), tile_sources in mp.tile_sources[b]
        if target_s in tile_sources
            relevant_sources = union(relevant_sources, tile_sources);
        end
    end

    relevant_sources
end


"""
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

end  # module

