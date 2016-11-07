# Routines for observations of light sources in particular images,
# rather than for sources in the abstract, in physical units,
# and rather than for images alone (that's image_model.jl).


"""
Attributes of the patch of sky surrounding a single
celestial object in a single image.

Attributes:
  - center: The approximate source location in world coordinates
  - radius_pix: The width of the influence of the object in pixel coordinates
  - psf: The point spread function in this region of the sky
  - wcs_jacobian: The jacobian of the WCS transform in this region of the
                  sky for each band
  - pixel_center: The pixel location of center in each band.
"""
immutable SkyPatch
    center::Vector{Float64}
    radius_pix::Float64

    psf::Vector{PsfComponent}
    wcs_jacobian::Matrix{Float64}
    pixel_center::Vector{Float64}

    center_int::Vector{Int64}
    radius_int::Int64
    active_pixel_bitmap::Matrix{Bool}
end


"""
Initialize a SkyPatch from an existing SkyPatch and a new PSF.
"""
SkyPatch(patch::SkyPatch, psf::Vector{PsfComponent}) =
    SkyPatch(patch.center, patch.radius_pix, psf, patch.wcs_jacobian,
             patch.pixel_center)


function choose_patch_radius(ce::CatalogEntry,
                             b::Int64,
                             psf_width::AbstractFloat,
                             epsilon::AbstractFloat;
                             width_scale=1.0)
    # The galaxy scale is the point with half the light -- if the light
    # were entirely in a univariate normal, this would be at 0.67 standard
    # deviations.  We are being a bit conservative here.
    obj_width =
      ce.is_star ? psf_width: width_scale * ce.gal_scale / 0.67 + psf_width

    flux = ce.is_star ? ce.star_fluxes[b] : ce.gal_fluxes[b]
    @assert flux > 0.

    # Choose enough pixels that the light is either 90% of the light
    # would be captured from a 1d gaussian or 5% of the sky noise,
    # whichever is a larger radius.
    pdf_90 = exp(-0.5 * (1.64)^2) / (sqrt(2pi) * obj_width)
    pdf_target = min(pdf_90, epsilon / (20 * flux))
    rhs = log(pdf_target) + 0.5 * log(2pi) + log(obj_width)
    sqrt(-2 * (obj_width ^ 2) * rhs)
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
    psf_width = Model.get_psf_width(psf, width_scale=width_scale)
    epsilon = img.epsilon_mat[round(Int, pixel_center[1]),
                              round(Int, pixel_center[2])]
    radius_req = choose_patch_radius(ce, img.b, psf_width, epsilon;
                                        width_scale=width_scale)
    min(radius_req, max_radius)
end


