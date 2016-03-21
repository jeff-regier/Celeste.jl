#!/usr/bin/env julia

using Celeste
using CelesteTypes

function load_cache(stamp_id)
    f = open(ENV["STAMP"]"/V-$stamp_id.dat")
    V = deserialize(f)
    close(f)
	V
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
	blob = Images.load_stamp_blob(ENV["STAMP"], stamp_id);

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
