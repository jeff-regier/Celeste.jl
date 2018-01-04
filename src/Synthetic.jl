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
    box = Model.box_around_point(img.wcs, ce.pos, 25)
    p = Model.ImagePatch(img, box)
    Model.write_star_nmgy!(ce.pos, ce.star_fluxes[img.b], p, img.pixels)
end


function write_galaxy_nmgy!(img::Image, ce::CatalogEntry)
    patches = Model.get_sky_patches([img], [ce], radius_override_pix=25.0)
    Model.write_galaxy_nmgy!(ce.pos, ce.gal_fluxes[img.b],
      ce.gal_frac_dev, ce.gal_axis_ratio, ce.gal_angle, ce.gal_radius_px,
      img.psf, patches, img.pixels)
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
function gen_images!(images::Vector{<:Image}, n_bodies::Vector{CatalogEntry}; expectation=false)
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
