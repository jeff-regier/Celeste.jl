module Synthetic

export gen_blob

using Celeste, Celeste.Model
import Celeste: Infer, ElboDeriv

import WCS
import Distributions

include("../src/bivariate_normals.jl")

# Generate synthetic data.

function wrapped_poisson(rate::Float64)
    0 < rate ? float(rand(Distributions.Poisson(rate))) : 0.
end


function get_patch(the_mean::Vector{Float64}, H::Int, W::Int)
    const radius = 50
    hm = round(Int, the_mean[1])
    wm = round(Int, the_mean[2])
    w11 = max(1, wm - radius):min(W, wm + radius)
    h11 = max(1, hm - radius):min(H, hm + radius)
    return(w11, h11)
end


function write_gaussian(the_mean, the_cov, intensity, pixels;
                        expectation=false)
    the_precision = the_cov^-1
    c = det(the_precision)^.5 / 2pi
    y = Array(Float64, 2)

    H, W = size(pixels)
    w_range, h_range = get_patch(the_mean, H, W)

    function matvec222(mat::Matrix, vec::Vector)
        # x' A x in a slightly more efficient form.
        (mat[1,1] * vec[1] + mat[1,2] * vec[2]) * vec[1] +
                (mat[2,1] * vec[1] + mat[2,2] * vec[2]) * vec[2]
    end

    for w in w_range, h in h_range
        y[1] = the_mean[1] - h
        y[2] = the_mean[2] - w
        ypy = matvec222(the_precision, y)
        pdf_hw = c * exp(-0.5 * ypy)
        pixel_rate = intensity * pdf_hw
        pixels[h, w] += expectation ? pixel_rate : wrapped_poisson(pixel_rate)
    end

    pixels
end


function write_star(img0::Image, ce::CatalogEntry, pixels::Matrix{Float64};
                    expectation=false)
    iota = median(img0.iota_vec)
    for k in 1:length(img0.psf)
        the_mean = WCS.world_to_pix(img0.wcs, ce.pos) + img0.psf[k].xiBar
        the_cov = img0.psf[k].tauBar
        intensity = ce.star_fluxes[img0.b] * iota * img0.psf[k].alphaBar
        write_gaussian(the_mean, the_cov, intensity, pixels,
            expectation = expectation)
    end
end


function write_galaxy(img0::Image, ce::CatalogEntry, pixels::Matrix{Float64};
                      expectation=false)
    iota = median(img0.iota_vec)
    e_devs = [ce.gal_frac_dev, 1 - ce.gal_frac_dev]
    XiXi = ElboDeriv.get_bvn_cov(ce.gal_ab, ce.gal_angle, ce.gal_scale)

    for i in 1:2
        for gproto in galaxy_prototypes[i]
            for k in 1:length(img0.psf)
                the_mean = WCS.world_to_pix(img0.wcs, ce.pos) +
                           img0.psf[k].xiBar
                the_cov = img0.psf[k].tauBar + gproto.nuBar * XiXi
                intensity = ce.gal_fluxes[img0.b] * iota *
                    img0.psf[k].alphaBar * e_devs[i] * gproto.etaBar
                write_gaussian(the_mean, the_cov, intensity, pixels,
                    expectation=expectation)
            end
        end
    end
end

function gen_image(img0::Image, n_bodies::Vector{CatalogEntry}; expectation=false)
    epsilon = img0.epsilon_mat[1]
    iota = img0.iota_vec[1]

    if expectation
        pixels = [epsilon * iota for h=1:img0.H, w=1:img0.W]
    else
        pixels = reshape(float(rand(Distributions.Poisson(epsilon * iota),
                         img0.H * img0.W)), img0.H, img0.W)
    end

    for body in n_bodies
        body.is_star ? write_star(img0, body, pixels) : write_galaxy(img0, body, pixels)
    end

    epsilon_mat = fill(epsilon, img0.H, img0.W)
    iota_vec = fill(iota, img0.H)

    return Image(img0.H, img0.W, pixels, img0.b, img0.wcs,
                 img0.psf,
                 img0.run_num, img0.camcol_num, img0.field_num,
                 epsilon_mat, iota_vec,
                 img0.raw_psf_comp)
end

"""
Generate a simulated blob based on a vector of catalog entries using
identity world coordinates.
"""
function gen_blob(blob0::Vector{Image}, n_bodies::Vector{CatalogEntry}; expectation=false)
    [gen_image(blob0[b], n_bodies, expectation=expectation) for b in 1:5]
end


#######################################

const pp = Model.load_prior()


function sample_fluxes(i::Int, r_s)
#    r_s = rand(Distributions.Normal(pp.r_mean[i], pp.r_var[i]))
    k_s = rand(Distributions.Categorical(pp.k[i]))
    c_s = rand(Distributions.MvNormal(pp.c[i][:, k_s], pp.c[i][:, :, k_s]))

    l_s = Array(Float64, 5)
    l_s[3] = r_s
    l_s[4] = l_s[3] * exp(c_s[3])
    l_s[5] = l_s[4] * exp(c_s[4])
    l_s[2] = l_s[3] / exp(c_s[2])
    l_s[1] = l_s[2] / exp(c_s[1])
    l_s
end


function synthetic_body(ce::CatalogEntry)
    ce2 = deepcopy(ce)
#    ce2.is_star = rand(Distributions.Bernoulli(pp.a[1]))
    ce2.star_fluxes[:] = sample_fluxes(1, ce.star_fluxes[3])
    ce2.gal_fluxes[:] = sample_fluxes(2, ce.gal_fluxes[3])
    ce2
end


function synthetic_bodies(n_bodies::Vector{CatalogEntry})
    CatalogEntry[synthetic_body(ce) for ce in n_bodies]
end


end
