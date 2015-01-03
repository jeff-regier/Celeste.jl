#!/usr/bin/env julia

import ElboDeriv
import OptimizeElbo
import StampBlob

using CelesteTypes
import ModelInit
using FITSIO
using WCSLIB


kelvin_ids = [
    "246.4702-19.0669", #8300 (flatish >5800)
    "310.4779-57.5232", #8850
    "240.0940-37.4041", #4650 (5400 and 5900 close)
    "270.0000-0.0030", #no spectrograph
    "150.5244--0.4788", #4800
    "118.3707-52.5271", #4600
    "305.1114--12.9577", #4000
    "130.1765-52.7501", #4000
    "93.9717-0.5630", #3950
    "253.1147-11.6072", #3900
    "188.3444-63.4421", #4050
]

peaks = Float64[8300, 8850, 4650, -1e20, 4800, 4600, 4000, 4000, 3950, 3900, 4050]
peak_ts = 2.897 * 1e7 ./ peaks

include("stamp_ids.jl")


type CatalogItem
	ra::Float64
	dec::Float64
	fluxes::Vector
	x::Float64
	y::Float64
end


function average_distance(a, b)
	ret = 0.
	count = 0
	for i in 1:size(a)[2]
		dist = minimum([norm(a[:, i] - b[:, j]) for j in 1:size(b)[2]])
		if dist < 1.1
			ret += dist
			count += 1
		end
	end
#	assert(count == min(size(a)[2], size(b)[2]))
	ret / count
end


function load_catalog(stamp_id)
	blob = StampBlob.load_stamp_blob(ENV["STAMP"], stamp_id);

	cat = fits_open_table(ENV["STAMP"]"/cat-$stamp_id.fits")
	num_rows = int(fits_read_keyword(cat, "NAXIS2")[1])
	table = Array(Float64, num_rows, 7)
	for i in 1:7
		data = Array(Float64, num_rows)
		fits_read_col(cat, Float64, i, 1, 1, data)
		table[:, i] = data
	end
	fits_close_file(cat)

	function row_to_cs(row)
		x_y = wcss2p(blob[1].wcs, row[1:2]'')
		CatalogItem(row[1], row[2], reverse(row[3:7][:]), x_y[1], x_y[2])
	end
	cat_stars = [row_to_cs(table[i, :][:]) for i in 1:num_rows]

	function in_bounds(cs)
		cs.x >= 3 && cs.x <= 49 && cs.y >= 3 && cs.y <= 49
	end
	cat_stars_2 = filter(in_bounds, cat_stars)

	cat_xy = Array(Float64, 2, length(cat_stars_2))
	for i in 1:length(cat_stars_2)
		cs = cat_stars_2[i]
		cat_xy[:, i] = [cs.x, cs.y]
	end

	cat_stars_2, cat_xy
end


function infer_and_cache(stamp_id)
	blob = StampBlob.load_stamp_blob(ENV["STAMP"], stamp_id);
	mp = ModelInit.peak_init(blob);

	OptimizeElbo.maximize_elbo(blob, M, V_init)

    f = open(ENV["STAMP"]"/V-$stamp_id.dat", "w+")
    serialize(f, V_opt)
    close(f)
end


function infer_and_cache()
	for stamp_id in stamp_ids
		infer_and_cache(stamp_id)
	end
end


function load_cache(stamp_id)
    f = open(ENV["STAMP"]"/V-$stamp_id.dat")
    V = deserialize(f)
    close(f)
	V
end


function center_star_id(V)
	distances = [norm(V[s].mu .- 51/2) for s in 1:length(V)]
	findmin(distances)[2]
end


function score_cached(stamp_id)
	star_cat, cat_xy = load_catalog(stamp_id)

	V = load_cache(stamp_id)

	println("found ", length(V), " out of ", length(star_cat), " stars")

	center_id = center_star_id(V)
	center_loc = real(V[center_id].mu)
	@printf("center star's location: [%.1f, %.1f]\n", center_loc[1], center_loc[2])

	starts = Array(Float64, 2, length(V))
	v1 = Array(Float64, 2, length(V))
	for s in 1:length(V)
		starts[:, s] = V[s].start
		v1[:, s] = V[s].mu
	end

	if min(length(V), length(star_cat)) > 0 && 25.5 < center_loc[1] < 26.5  && 25.5 < center_loc[2] < 26.5 
		@printf("location error (pixels): %.2f (at initialization), %.2f (predicted)\n", 
			average_distance(starts, cat_xy),
			average_distance(v1, cat_xy))

		fits = FITS(ENV["STAMP"]"/stamp-r-$stamp_id.fits")
		hdr = readheader(fits[1])
		@printf("temperature (Kelvin): %.0f (predicted), %.0f (actual)\n",
			real(V[center_id].tau), 
			hdr["T_EFF"])
		close(fits)
	end
end


function score_cached()
	for stamp_id in stamp_ids
		println("-----------------------------------------------")
		println(stamp_id)
		score_cached(stamp_id)
	end
end


function temperature_correlation()
	predicted_t = Float64[]
	true_t = Float64[]
	for stamp_id in stamp_ids
		V = load_cached(stamp_id)

		fits = FITS(ENV["STAMP"]"/stamp-r-$stamp_id.fits")
		hdr = readheader(fits[1])

		push!(predicted_t, real(V.t1[center_star_id(V.v1)]))
		push!(true_t, hdr["T_EFF"])
	end

	using Winston
	plot(predicted_t, true_t)

	using DataFrames
	df = DataFrame()
	df[:y] = true_t
	df[:x] = predicted_t

	using GLM
#	glm(y~x,df,Normal(),IdentityLink())
end


function location_error()
	start_err = 0.
	v1_err = 0.
	num_stars = 0
	for stamp_id in stamp_ids
		println(stamp_id)

		V = load_cached(stamp_id)
		star_cat, cat_xy = load_catalog(stamp_id)

		count = min(length(V), length(star_cat))
		if count > 0
			num_stars += count
			start_err += count * average_distance(V.starts, cat_xy)
			v1_err += count * average_distance(V.v1, cat_xy)
		end
	end

	(start_err / num_stars, v1_err / num_stars)
end


function score_peaks(stamp_id)
	blob = StampBlob.load_stamp_blob(ENV["STAMP"], stamp_id);
	M = sample_prior();
	V = init_sources(blob);

	star_cat, cat_xy = load_catalog(stamp_id)

	println("found ", size(V.v1)[2], " out of ", length(star_cat), " stars")
end


function score_peaks()
	for stamp_id in stamp_ids
		println("-----------------------------------------------")
		println(stamp_id)
		score_peaks(stamp_id)
	end
end


function ef_pixels(img::Image, V::VariationalParams)
	E_F = Array(Float64, size(img.pixels))
	fill!(E_F, img.epsilon)

	wimg = Elbo.WorkingImage(img, V)
	for is in wimg.intermediate_sources
        for k in 1:3
            (w_range, h_ranges) = is.patches[k]
            for w in w_range, h in h_ranges[w - w_range[1] + 1]
                pdf_hw = Elbo.pdf(is.mvns[k], Float64[h, w])
                E_F[h, w] += pdf_hw * img.alphaBar[k] * is.E_bI
            end
        end
	end

	E_F
end

function posterior_check_plot(stamp_id)
	blob = StampBlob.load_stamp_blob(ENV["STAMP"], stamp_id);

	V = load_cache(stamp_id)

	raw = blob[3].pixels
	E_F = ef_pixels(blob[3], V)
	vmax = maximum(vcat(raw, E_F))

	import PyPlot
	fig, axes = PyPlot.subplots(nrows=1, ncols=2)

	PyPlot.subplot(1, 2, 1)
	sp1 = PyPlot.imshow(raw, interpolation="nearest", vmin=0, vmax=vmax)
	PyPlot.colorbar(sp1, shrink=0.4)

	PyPlot.subplot(1, 2, 2)
	sp2 = PyPlot.imshow(E_F, interpolation="nearest", vmin=0, vmax=vmax)
	PyPlot.colorbar(sp2, shrink=0.4)

	PyPlot.savefig(ENV["STAMP"]"/plot-$stamp_id.png")
end


function posterior_check_plot()
	for stamp_id in stamp_ids
		println("-----------------------------------------------")
		println(stamp_id)
		posterior_check_plot(stamp_id)
	end
end



