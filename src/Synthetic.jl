module Synthetic

export gen_image!, gen_images!

using Celeste
using Celeste.Model
using Celeste.SensitiveFloats

import WCS
import Distributions
using ForwardDiff
using StaticArrays


# Generate synthetic data.

function wrapped_poisson(rate::Float64)
    0 < rate ? float(rand(Distributions.Poisson(rate))) : 0.
end


function get_patch(the_mean::SVector{2,Float64}, H::Int, W::Int)
    radius = 50
    hm = round(Int, the_mean[1])
    wm = round(Int, the_mean[2])
    w11 = max(1, wm - radius):min(W, wm + radius)
    h11 = max(1, hm - radius):min(H, hm + radius)
    return (w11, h11)
end


function write_gaussian!(pixel_rates, the_mean, the_cov, intensity)
    the_precision = inv(the_cov)
    c = sqrt(det(the_precision)) / 2pi

    H, W = size(pixel_rates)
    w_range, h_range = get_patch(the_mean, H, W)

    for w in w_range, h in h_range
        y = @SVector [the_mean[1] - h, the_mean[2] - w] # Maybe not hard code Float64
        ypy = dot(y,  the_precision * y)
        pdf_hw = c * exp(-0.5 * ypy)
        pixel_rates[h, w] += intensity * pdf_hw
    end
end


function write_star_nmgy!(img::Image, ce::CatalogEntry)
    p = Model.SkyPatch(img, ce)
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
    gal_frac_devs = [ce.gal_frac_dev, 1 - ce.gal_frac_dev]
    XiXi = Model.get_bvn_cov(ce.gal_axis_ratio, ce.gal_angle, ce.gal_radius_px)

    for i in 1:2
        for gproto in galaxy_prototypes[i]
            for k in 1:length(img.psf)
                the_mean = SVector{2}(WCS.world_to_pix(img.wcs, ce.pos)) +
                           img.psf[k].xiBar
                the_cov = img.psf[k].tauBar + gproto.nuBar * XiXi
                intensity = ce.gal_fluxes[img.b] *
                    img.psf[k].alphaBar * gal_frac_devs[i] * gproto.etaBar
                write_gaussian!(img.pixels, the_mean, the_cov, intensity)
            end
        end
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
