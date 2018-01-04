using Interpolations

# Routines for observations of light sources in particular images,
# rather than for sources in the abstract, in physical units,
# and rather than for images alone (that's image_model.jl).

# A contiguous box in a 2-d array.
const Box = Tuple{UnitRange{Int}, UnitRange{Int}}

clamp_box(box::Box, dims::Tuple{Int, Int}) =
    (clamp(first(box[1]), 1, dims[1]+1):
     clamp(last(box[1]), 0, dims[1]),
     clamp(first(box[2]), 1, dims[2]+1):
     clamp(last(box[2]), 0, dims[2]))

function _dilate_range(range::UnitRange{Int}, factor::Float64)
    delta = round(Int, factor * length(range) / 2)
    return (first(range) - delta):(last(range) + delta)
end

dilate_box(box::Box, factor::Float64) = (_dilate_range(box[1], factor),
                                         _dilate_range(box[2], factor))

_enclose_ranges(range1::UnitRange{Int}, range2::UnitRange{Int}) =
    min(first(range1), first(range2)):max(last(range1), last(range2))

enclose_boxes(box1::Box, box2::Box) = (_enclose_ranges(box1[1], box2[1]),
                                       _enclose_ranges(box1[2], box2[2]))

_intersect_ranges(range1::UnitRange{Int}, range2::UnitRange{Int}) =
    max(first(range1), first(range2)):min(last(range1), last(range2))

intersect_boxes(box1::Box, box2::Box) = (_intersect_ranges(box1[1], box2[1]),
                                         _intersect_ranges(box1[2], box2[2]))

_ranges_overlap(range1::UnitRange{Int}, range2::UnitRange{Int}) =
    (first(range1) <= last(range2)) && (first(range2) <= last(range1))

boxes_overlap(box1::Box, box2::Box) = (_ranges_overlap(box1[1], box2[1]) &&
                                       _ranges_overlap(box1[2], box2[2]))


"""
    ImagePatch(img::Image, ce:CatalogEntry; radius_override_pix=NaN)

Attributes of a subsection of an Image around a point of interest.

# Attributes:
- `box`: The pixel range in the image.
- `psf`: The point spread function in the center of the image
- `itp_psf`: The point spread function in this region
- `wcs_jacobian`: The jacobian of the WCS transform in this region of the
                  sky for each band.
- `world_center`: A reference position in world coordinates.
- `pixel_center`: A reference position in pixel coordinates.
- `bitmap_offset`: Lower left corner index offset.
- `active_pixel_bitmap`: Boolean mask denoting which pixels in the patch are
                         considered when processing the source.
"""
struct ImagePatch
    box::Box
    world_center::Vector{Float64}

    psf::Vector{PsfComponent}
    itp_psf::AbstractInterpolation
    wcs_jacobian::Matrix{Float64}
    pixel_center::Vector{Float64}

    bitmap_offset::SVector{2, Int64}  # lower left corner index offset
    active_pixel_bitmap::Matrix{Bool}
end


function Base.show(io::IO, patch::ImagePatch)
    print(io, "$(length(patch.box[1]))×$(length(patch.box[2])) ImagePatch at $(patch.box)")
end


# construct `ImagePatch` from box of pixels on image.
function ImagePatch(img::Image, box::Box)
    # Crop off-image portion of box. Completely off-image boxes are
    # allowed and internally indicated with a range of 1:0 or H+1:H
    # (an empty range, but still a legal index to image pixels).
    box = clamp_box(box, (img.H, img.W))

    # Get linear WCS transform at center of box.
    pixel_center = [(first(box[1]) + last(box[1])) / 2
                    (first(box[2]) + last(box[2])) / 2]
    world_center = WCS.pix_to_world(img.wcs, pixel_center)
    wcs_jacobian = pixel_world_jacobian(img.wcs, pixel_center)

    # active pixel bitmap: make masked (NaN) pixels non-active
    box_offset = SVector(first(box[1]) - 1, first(box[2]) - 1)
    active_pixel_bitmap = [!isnan(img.pixels[x, y])
                           for x in box[1], y in box[2]]

    grid_psf = img.psfmap(pixel_center[1], pixel_center[2])
    grid_psf[:, :] = max.(grid_psf, 0.0)
    grid_psf += 1e-6
    grid_psf /= sum(grid_psf)
    # The following transformation is like softplus. Its inv always returns a
    # positive value. Without this transformation, even if the psf over the
    # grid_psf is positive, the iterpolation of grid with bicubic splines often
    # has negative values.
    grid_psf[:, :] = softpluslike.(grid_psf)

    itp_psf = interpolate(grid_psf, BSpline(Cubic(Line())), OnGrid())

    ImagePatch(box,
             world_center,
             img.psf,
             itp_psf,
             wcs_jacobian,
             pixel_center,
             box_offset,
             active_pixel_bitmap)
end


"""
    box_around_point(wcs, world_center, box_radius)

Choose patch indexes given center world coordinates and a box "radius" in
pixels.
"""
function box_around_point(wcs::WCS.WCSTransform, world_center::Vector{Float64},
                          pixel_radius::Real)
    pixel_center = WCS.world_to_pix(wcs, world_center)

    xmin = round(Int, pixel_center[1] - pixel_radius)
    xmax = round(Int, pixel_center[1] + pixel_radius)
    ymin = round(Int, pixel_center[2] - pixel_radius)
    ymax = round(Int, pixel_center[2] + pixel_radius)

    return (xmin:xmax, ymin:ymax)
end



"""
    box_from_catalog(img::Image, ce::CatalogEntry, scale)

Return a box extent based on `CatalogEntry` and `Image` parameters,
optionally scaled by a multiplicative factor `scale`. This is
useful when initialing an `ImagePatch` from a catalog rather than detections.
"""
function box_from_catalog(img::Image, ce::CatalogEntry;
                          width_scale=1.0, max_radius=25)

    pixel_radius = choose_patch_radius(ce, img; width_scale=width_scale,
                                       max_radius=max_radius)
    pixel_center = WCS.world_to_pix(img.wcs, ce.pos)

    xmin = round(Int, pixel_center[1] - pixel_radius)
    xmax = round(Int, pixel_center[1] + pixel_radius)
    ymin = round(Int, pixel_center[2] - pixel_radius)
    ymax = round(Int, pixel_center[2] + pixel_radius)

    return (xmin:xmax, ymin:ymax)
end




function get_sky_patches(images::Vector{<:Image},
                         catalog::Vector{CatalogEntry};
                         radius_override_pix=NaN)
    N = length(images)
    S = length(catalog)
    patches = Matrix{ImagePatch}(S, N)

    for n in 1:N, s in 1:S
        box = (isnan(radius_override_pix)?
               box_from_catalog(images[n], catalog[s]; width_scale=1.2):
               box_around_point(images[n].wcs, catalog[s].pos,
                                radius_override_pix))
        pixel_center = WCS.world_to_pix(images[n].wcs, catalog[s].pos)
        patches[s, n] = ImagePatch(images[n], box)
    end

    patches
end


"""
Choose a reasonable patch radius based on the catalog.

Args:
  - ce: The catalog entry for the object
  - img: The image
  - width_scale: A multiple of standard deviations to use
  - max_radius: The maximum radius in pixels.

Returns:
  - A radius in pixels chosen from the catalog entry.
"""
function choose_patch_radius(ce::CatalogEntry,
                             img::Image;
                             width_scale=1.0,
                             max_radius=25)
    psf_width = Model.get_psf_width(img.psf, width_scale=width_scale)

    # The galaxy scale is the point with half the light -- if the light
    # were entirely in a univariate normal, this would be at 0.67 standard
    # deviations.  We are being a bit conservative here.
    obj_width = ce.is_star ? 0.0 : width_scale * ce.gal_radius_px / 0.67
    obj_width += psf_width

    flux = ce.is_star ? ce.star_fluxes[img.b] : ce.gal_fluxes[img.b]
    if (!(flux > 0.)) @show ce end
    @assert flux > 0.

    # Choose enough pixels that the light is either 80% of the light
    # would be captured from a 1d gaussian or 5% of the sky noise,
    # whichever is a larger radius.
    epsilon = img.sky[div(img.H, 2), div(img.W, 2)]
    pdf_90 = exp(-0.5 * (1.64)^2) / (sqrt(2pi) * obj_width)
    pdf_target = min(pdf_90, epsilon / (20 * flux))
    rhs = log(pdf_target) + 0.5 * log(2pi) + log(obj_width)
    radius_req = sqrt(-2 * (obj_width ^ 2) * rhs)

    min(radius_req, max_radius)
end

"""
    find_neighbors(patches, target)

Return indexes of objects in `patches` whose boxes overlap the object
at index `target` in any image. (The first and second axes of `patches`
are over objects and images, respectively.)
"""
function find_neighbors(patches::Matrix{ImagePatch}, target::Int)
    neighbors = Int[]
    for i in 1:size(patches, 1)  # loop over objects
        i == target && continue
        for j in 1:size(patches, 2)  # loop over images
            if boxes_overlap(patches[target, j].box, patches[i, j].box)
                push!(neighbors, i)
                break
            end
        end
    end
    return neighbors
end
