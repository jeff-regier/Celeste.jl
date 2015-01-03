# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module ViInit

export sample_prior, init_sources

using FITSIO
using Distributions
using WCSLIB
using Util
using CelesteTypes


function sample_prior()
	rho = 0.5
	Delta = eye(4) * 1e-2

	const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")
	f = open("$dat_dir/ThetaLambda.dat")
	(Theta, Lambda) = deserialize(f)
	close(f)

	PriorParams(rho, Delta, Theta, Lambda)
end


init_source(init_pos::Vector{Float64}) = begin
	ParamStruct{Float64}(
		0.5,
		(init_pos[1], init_pos[2]),
		(50_000, 50_000,  50_000,  50_000,  50_000), 
		5000.,
		(50_000, 50_000,  50_000,  50_000,  50_000), 
		0.5,
		(1.5, 0., 1.5))
end


function matched_filter(img::Image)
	H, W = 5, 5
	kernel = zeros(Float64, H, W)
	for k in 1:3
		mvn = MvNormal(img.psf[k].xiBar, img.psf[k].SigmaBar)
		for h in 1:H
			for w in 1:W
				x = [h - (H + 1) / 2., w - (W + 1) / 2.]
				kernel[h, w] += img.psf[k].alphaBar * pdf(mvn, x)
			end
		end
	end
	kernel /= sum(kernel)
end


function convolve_image(img::Image)
	kernel = matched_filter(img)
	H, W = size(img.pixels)
	padded_pixels = Array(Float64, H + 8, W + 8)
	fill!(padded_pixels, median(img.pixels))
	padded_pixels[5:H+4,5:W+4] = img.pixels
	conv2(padded_pixels, kernel)[7:H+6, 7:W+6]
end


function peak_starts(blob::Blob)
	H, W = size(blob[1].pixels)
	added_pixels = zeros(Float64, H, W)
	for b in 1:5
		added_pixels += convolve_image(blob[b])
	end
	spread = quantile(added_pixels[:], .7) - quantile(added_pixels[:], .2)
	threshold = median(added_pixels) + 3spread

	peaks = Array(Vector{Float64}, 0)
	i = 0
	for h=3:(H-3), w=3:(W-3)
		if added_pixels[h, w] > threshold &&
				added_pixels[h, w] > maximum(added_pixels[h-2:h+2, w-2:w+2]) - .1
			i += 1
#			println("found peak $i: ", h, " ", w)
#			println(added_pixels[h-3:min(h+3,99), w-3:min(w+3,99)])
			push!(peaks, [h, w])
		end
	end

	R = length(peaks)
	peaks_mat = Array(Float64, 2, R)
	for i in 1:R
		peaks_mat[:, i] = peaks[i]
	end

	peaks_mat
#	wcsp2s(img.wcs, peaks_mat)
end


function init_sources(blob::Blob)
	v1 = peak_starts(blob)
	S = size(v1)[2]
	[init_source(v1[:, s]) for s in 1:S]
end


end
