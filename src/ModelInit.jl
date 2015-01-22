# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module ModelInit

export sample_prior, cat_init, peak_init

using FITSIO
using Distributions
using WCSLIB
using Util
using CelesteTypes


function sample_prior()
	Delta = 0.5

	const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

	Upsilon = Array(Float64, 2)
	Phi = Array(Float64, 2)
	r_file = open("$dat_dir/r_prior.dat")
	((Upsilon[1], Phi[1]), (Upsilon[2], Phi[2])) = deserialize(r_file)
	close(r_file)

	Psi = Array(Vector{Float64}, 2)
	Omega = Array(Array{Float64, 2}, 2)
	Lambda = Array(Array{Array{Float64, 2}}, 2)
	ck_file = open("$dat_dir/ck_prior.dat")
	((Psi[1], Omega[1], Lambda[1]), (Psi[2], Omega[2], Lambda[2])) = deserialize(ck_file)
	close(r_file)

	PriorParams(Delta, Upsilon, Phi, Psi, Omega, Lambda)
end


#TODO: use blob (and perhaps priors) to initialize these sensibly
function init_source(init_pos::Vector{Float64})
	ret = Array(Float64, length(all_params))
	ret[ids.chi] = 0.5
	ret[ids.mu[1]] = init_pos[1]
	ret[ids.mu[2]] = init_pos[2]
	ret[ids.gamma] = 1e5
	ret[ids.zeta] = 1e-4
	ret[ids.theta] = 0.5
	ret[ids.Xi[1]] = ret[ids.Xi[3]] = 1.5
	ret[ids.Xi[2]] = 0.
	ret[ids.kappa] = 1. / size(ids.kappa, 1)
	ret[ids.beta] = 0.
	ret[ids.lambda] =  1e-4
	ret
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

	peaks_mat - .5  #e.g. the center of pixel [1,1] is [0.5, 0.5]
#	wcsp2s(img.wcs, peaks_mat)
end


function peak_init(blob::Blob; patch_radius::Float64=Inf,
		tile_width::Int64=typemax(Int64))
	v1 = peak_starts(blob)
	S = size(v1)[2]
	vp = [init_source(v1[:, s]) for s in 1:S]
	twice_radius = float(max(blob[1].H, blob[1].W))
	# TODO: use non-trival patch radii, based on blob detection routine
	patches = [SkyPatch(v1[:, s], patch_radius) for s in 1:S]
	ModelParams(vp, sample_prior(), patches, tile_width)
end


function cat_init(cat::Vector{CatalogEntry}; patch_radius::Float64=Inf,
		tile_width::Int64=typemax(Int64))
	vp = [init_source(ce.pos) for ce in cat]
	# TODO: use non-trivial patch radii, based on the catalog
	patches = [SkyPatch(ce.pos, patch_radius) for ce in cat]
	ModelParams(vp, sample_prior(), patches, tile_width)
end


end
