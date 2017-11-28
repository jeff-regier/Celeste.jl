using Interpolations

# Routines for observations of light sources in particular images,
# rather than for sources in the abstract, in physical units,
# and rather than for images alone (that's image_model.jl).

# A contiguous box in a 2-d array.
const Box = Tuple{UnitRange{Int}, UnitRange{Int}}


"""
    SkyPatch(img::Image, ce:CatalogEntry; radius_override_pix=NaN)

Attributes of the patch of sky surrounding a single celestial object `ce`
in a single image `img`. If `radius_override_pix` is not NaN, it will be
used for the `radius_pix` attribute.

# Attributes:
- `center`: The approximate source location in world coordinates
- `radius_pix`: The width of the influence of the object in pixel coordinates
- `psf`: The point spread function in this region of the sky
- `itp_psf`
- `wcs_jacobian`: The jacobian of the WCS transform in this region of the
                  sky for each band.
- `pixel_center`: The pixel location of center in each band.
- `bitmap_offset`: Lower left corner index offset.
- `active_pixel_bitmap`: Boolean mask denoting which pixels in the patch are
                         considered when processing the source.
"""
struct SkyPatch
    world_center::Vector{Float64}
    radius_pix::Float64

    psf::Vector{PsfComponent}
    itp_psf::AbstractInterpolation
    wcs_jacobian::Matrix{Float64}
    pixel_center::Vector{Float64}

    bitmap_offset::SVector{2, Int64}  # lower left corner index offset
    active_pixel_bitmap::Matrix{Bool}
end


function SkyPatch(img::Image, ce::CatalogEntry; radius_override_pix=NaN)
    world_center = ce.pos
    pixel_center = WCS.world_to_pix(img.wcs, world_center)
    wcs_jacobian = pixel_world_jacobian(img.wcs, pixel_center)
    radius_pix = choose_patch_radius(ce, img, width_scale=1.2)
    @assert radius_pix <= 25
    if !isnan(radius_override_pix)
        radius_pix = radius_override_pix
    end

    hmin = max(0, floor(Int, pixel_center[1] - radius_pix - 1))
    hmax = min(img.H - 1, ceil(Int, pixel_center[1] + radius_pix - 1))
    wmin = max(0, floor(Int, pixel_center[2] - radius_pix - 1))
    wmax = min(img.W - 1, ceil(Int, pixel_center[2] + radius_pix - 1))

    # some light sources are so far from some images that they don't
    # overlap at all
    H2 = max(0, hmax - hmin + 1)
    W2 = max(0, wmax - wmin + 1)

    # all pixels are active by default
    active_pixel_bitmap = trues(H2, W2)

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

    SkyPatch(world_center,
             radius_pix,
             img.psf,
             itp_psf,
             wcs_jacobian,
             pixel_center,
             SVector(hmin, wmin),
             active_pixel_bitmap)
end


# construct `SkyPatch` from box of pixels on image.
function SkyPatch(img::Image, box::Box)
    # Crop off-image portion of box. Completely off-image boxes are
    # allowed and internally indicated with a range of 1:0 or H+1:H
    # (an empty range, but still a legal index to image pixels).
    box = (clamp(first(box[1]), 1, img.H+1):
           clamp(last(box[1]), 0, img.H),
           clamp(first(box[2]), 1, img.W+1):
           clamp(last(box[2]), 0, img.W))

    # Get linear WCS transform at center of box.
    pixel_center = [(first(box[1]) + last(box[1])) / 2
                    (first(box[2]) + last(box[2])) / 2]
    world_center = pix_to_world(img.wcs, pixel_center)
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

    SkyPatch(world_center,
             0.0,  # not used
             img.psf,
             itp_psf,
             wcs_jacobian,
             pixel_center,
             SVector(hmin, wmin),
             active_pixel_bitmap)
end


function get_sky_patches(images::Vector{<:Image},
                         catalog::Vector{CatalogEntry};
                         radius_override_pix=NaN)
    N = length(images)
    S = length(catalog)
    patches = Matrix{SkyPatch}(S, N)

    for n in 1:N, s in 1:S
        patches[s, n] = SkyPatch(images[n],
                                 catalog[s],
                                 radius_override_pix=radius_override_pix)
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
