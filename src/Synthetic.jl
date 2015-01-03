module Synthetic

export StarParams, GalaxyParams, gen_blob

using CelesteTypes
import Util

import Distributions


function wrapped_poisson(rate::Float64)
    0 < rate ? float(rand(Distributions.Poisson(rate))) : 0.
end


abstract BodyParams

type StarParams <: BodyParams
	mu::Vector{Float64}
	gamma::Vector{Float64}
end

type GalaxyParams <: BodyParams
    mu::Vector{Float64}
    zeta::Vector{Float64}
    theta::Float64
    Xi::Vector{Float64}
end 


function get_patch(the_mean::Vector{Float64}, H::Int64, W::Int64)
    const radius = 50.
    hm, wm = int(the_mean)
    w11 = max(1, wm - radius):min(W, wm + radius)
    h11 = max(1, hm - radius):min(H, hm + radius)
    return(w11, h11)
end


function write_gaussian(the_mean, the_cov, intensity, pixels)
    the_precision = the_cov^-1
    c = det(the_precision)^.5 / 2pi
	y = Array(Float64, 2)

	H, W = size(pixels)
	w_range, h_range = get_patch(the_mean, H, W)

	for w in w_range, h in h_range
		y[1] = the_mean[1] - h
		y[2] = the_mean[2] - w
		ypy = Util.matvec222(the_precision, y)
		pdf_hw = c * exp(-0.5 * ypy)
		pixel_rate = intensity * pdf_hw
		pixels[h, w] += wrapped_poisson(pixel_rate)
	end

	pixels
end


function write_body(img0::Image, sp::StarParams, pixels::Matrix{Float64})
	for k in 1:length(img0.psf)
		the_mean = sp.mu + img0.psf[k].xiBar
		the_cov = img0.psf[k].SigmaBar
		intensity = sp.gamma[img0.b] * img0.psf[k].alphaBar
		write_gaussian(the_mean, the_cov, intensity, pixels)
	end
end


function write_body(img0::Image, gp::GalaxyParams, pixels::Matrix{Float64})
	thetas = [gp.theta, 1 - gp.theta]

    Xi = [[gp.Xi[1] gp.Xi[2]], [0.  gp.Xi[3]]]
    XiXi = Xi' * Xi

	for i in 1:2
		for gproto in galaxy_prototypes[i]
			for k in 1:length(img0.psf)
				the_mean = gp.mu + img0.psf[k].xiBar
				the_cov = img0.psf[k].SigmaBar + gproto.sigmaTilde * XiXi
				intensity = gp.zeta[img0.b] * img0.psf[k].alphaBar * thetas[i] * gproto.alphaTilde
				write_gaussian(the_mean, the_cov, intensity, pixels)
			end
		end
	end
end


function gen_image(img0::Image, n_bodies::Vector{BodyParams})
    pixels = reshape(float(rand(Distributions.Poisson(img0.epsilon),
					 img0.H * img0.W)), img0.H, img0.W)

	for body in n_bodies
		write_body(img0, body, pixels)
	end

    return Image(img0.H, img0.W, pixels, img0.b, img0.wcs, img0.epsilon, img0.psf)
end


function gen_blob(blob0::Blob, n_bodies::Vector{BodyParams})
	[gen_image(blob0[b], n_bodies) for b in 1:5]
end


end

