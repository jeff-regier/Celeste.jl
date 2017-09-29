#  Generates synthetic data.
module Synthetic
export gen_image!, gen_images!

using Celeste
using Celeste.Model
using Celeste.SensitiveFloats

import WCS
import Distributions
using ForwardDiff
using StaticArrays


function write_star_nmgy!(img::Image, ce::CatalogEntry)
    p = Model.SkyPatch(img, ce, radius_override_pix=25)
    fs0m = SensitiveFloat{Float64}(length(StarPosParams), 1, false, false)

    H2, W2 = size(p.active_pixel_bitmap)
    for w2 in 1:W2, h2 in 1:H2
        h = p.bitmap_offset[1] + h2
        w = p.bitmap_offset[2] + w2
        Model.star_light_density!(fs0m, p, h, w, ce.pos, false)
        img.pixels[h, w] += fs0m.v[] * ce.star_fluxes[img.b]
    end
end


function write_galaxy_nmgy!(img::Image, ce::CatalogEntry)
    bvn_derivs = Model.BivariateNormalDerivatives{Float64}()
    fs1m = SensitiveFloat{Float64}(length(GalaxyPosParams), 1, false, false)
    patches = Model.get_sky_patches([img], [ce], radius_override_pix=25.0)
    source_params = [[ce.pos[1], ce.pos[2], ce.gal_frac_dev, ce.gal_axis_ratio,
                     ce.gal_angle, ce.gal_radius_px],]
    star_mcs, gal_mcs = Model.load_bvn_mixtures(1, patches,
                          source_params, [1,], length(img.psf), 1,
                          calculate_gradient=false,
                          calculate_hessian=false)
    p = patches[1]
    H2, W2 = size(p.active_pixel_bitmap)
    for w2 in 1:W2, h2 in 1:H2
        # (h2, w2) index the local patch, while (h, w) index the image
        h = p.bitmap_offset[1] + h2
        w = p.bitmap_offset[2] + w2
        Model.populate_gal_fsm!(fs1m, bvn_derivs, 1, h, w, false, p.wcs_jacobian, gal_mcs)
        img.pixels[h, w] += fs1m.v[] * ce.gal_fluxes[img.b]
    end
end


function gen_image!(img::Image, n_bodies::Vector{CatalogEntry}; expectation=false)
    H, W = size(img.pixels)
    for w in 1:W, h in 1:H
        img.pixels[h, w] = img.sky[h, w]  # in nmgy
    end

    for body in n_bodies
        body.is_star ? write_star_nmgy!(img, body) : write_galaxy_nmgy!(img, body)
    end

    img.pixels .*= img.nelec_per_nmgy

    if !expectation
        for w in 1:W, h in 1:H
            img.pixels[h, w] = rand(Distributions.Poisson(img.pixels[h, w]))
        end
    end
end


"""
Generate a synthetic images based on a vector of catalog entries using
identity world coordinates.
"""
function gen_images!(images::Vector{Image}, n_bodies::Vector{CatalogEntry}; expectation=false)
    for img in images
        gen_image!(img, n_bodies; expectation=expectation)
    end
end


#######################################

const pp = Model.load_prior()


function sample_fluxes(i::Int, r_s)
    k_s = rand(Distributions.Categorical(pp.k[i]))
    c_s = rand(Distributions.MvNormal(pp.color[i][:, k_s], pp.color[i][:, :, k_s]))

    l_s = Vector{Float64}(5)
    l_s[3] = r_s
    l_s[4] = l_s[3] * exp(c_s[3])
    l_s[5] = l_s[4] * exp(c_s[4])
    l_s[2] = l_s[3] / exp(c_s[2])
    l_s[1] = l_s[2] / exp(c_s[1])
    l_s
end


function synthetic_body(ce::CatalogEntry)
    ce2 = deepcopy(ce)
#    ce2.is_star = rand(Distributions.Bernoulli(pp.is_star[1]))
    ce2.star_fluxes[:] = sample_fluxes(1, ce.star_fluxes[3])
    ce2.gal_fluxes[:] = sample_fluxes(2, ce.gal_fluxes[3])
    ce2
end


function synthetic_bodies(n_bodies::Vector{CatalogEntry})
    CatalogEntry[synthetic_body(ce) for ce in n_bodies]
end


end
