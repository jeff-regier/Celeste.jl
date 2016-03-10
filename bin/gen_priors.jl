#!/usr/bin/env julia

# Note!  This requires GuassianMixtures, which is not in REQURE.

using FITSIO
using GaussianMixtures
using Distributions


function mag_to_nanomaggies(mag::Float64)
	10 ^ (( mag - 22.5 ) / -2.5)
end

@vectorize_1arg Float64 mag_to_nanomaggies


function read_r_colors(prior_file)
	fits = FITS(prior_file)
  cat = fits[2]
	num_rows, = read_key(cat, "NAXIS2")
	table = Array(Float64, num_rows, 12)
	for i in 1:12
    col_name, = read_key(cat, "TTYPE$i")
		table[:, i] = read(cat, col_name)
	end
	close(fits)

	t2 = table[:,:]
	for err_col in 8:12
		t2 = t2[t2[:,err_col] .< median(table[:,err_col]) / 2, :]
	end

	for b in 2:7
		t2 = t2[t2[:, b] .> -50, :]
	end

	t3 = mag_to_nanomaggies(t2[:, 3:7])
	colors = Array(Float64, size(t2)[1], 4)
	for i in 1:4
		colors[:, i] = log(t3[:, i + 1] ./ t3[:, i])
	end

	#TODO: convert 'r' band to nanomaggies
	colors, mag_to_nanomaggies(table[table[:, 5] .> 1, 5])
end


function vecmat_to_tensor(vecmat::Vector{Matrix{Float64}})
    ret = Array(Float64, size(vecmat[1], 1), size(vecmat[1], 2), length(vecmat))
    for i in 1:length(vecmat)
        ret[:, :, i] = vecmat[i]
    end
    ret
end


if length(ARGS) != 2
    println("usage: gen_priors.jl [catalog.fits] [out_file.dat]")
else
    c0, r0 = read_r_colors(ARGS[1])

    fit_r = fit_mle(LogNormal, r0)

    D = 64
    c0_train = c0
    #c0_test = c0[120001:end, :]
    fit_gmm = GMM(D, c0_train, kind=:full, method=:split)
    println("train avll:", GaussianMixtures.avll(fit_gmm, c0_train))
    #println("test avll:", GaussianMixtures.avll(fit_gmm, c0_test))

    out_file = open(ARGS[2], "w+")
    serialize(out_file, (
        params(fit_r),
        weights(fit_gmm),
        means(fit_gmm)',
        vecmat_to_tensor(covars(fit_gmm))))
    close(out_file)
end
