#!/usr/bin/env julia

# Note!  This requires GuassianMixtures, which is not in REQURE.

using FITSIO
using GaussianMixtures
using Distributions
using JLD


function mag_to_nanomaggies(mag::Float64)
	10 ^ (( mag - 22.5 ) / -2.5)
end


function read_r_colors(prior_file)
	fits = FITS(prior_file)
    cat = fits[2]
	num_rows, = read_key(cat, "NAXIS2")
	table = Matrix{Float64}(num_rows, 12)
	for i in 1:12
    col_name, = read_key(cat, "TTYPE$i")
		table[:, i] = read(cat, col_name)
	end
	close(fits)

	t2 = table[:,:]
	for b in 2:7
		t2 = t2[t2[:, b] .> -50, :]
	end

	t3 = mag_to_nanomaggies.(t2[:, 3:7])
	colors = Matrix{Float64}(size(t2, 1), 4)
	for i in 1:4
		colors[:, i] = log(t3[:, i + 1] ./ t3[:, i])
	end

	#TODO: convert 'r' band to nanomaggies
	colors, mag_to_nanomaggies.(table[table[:, 5] .> 1, 5])
end


function read_quasar_catalog(prior_file)
	cat = FITS(prior_file)[2]
    mags = read(cat, "PSFMAG")'
    log_nmgys = log.(mag_to_nanomaggies.(mags))
    colors = log_nmgys[:, 2:5] - log_nmgys[:, 1:4]
    r = exp.(log_nmgys[:, 3])
    colors, r
end

function vecmat_to_tensor(vecmat::Vector{Matrix{Float64}})
    ret = Array{Float64,3}(size(vecmat[1], 1), size(vecmat[1], 2), length(vecmat))
    for i in 1:length(vecmat)
        ret[:, :, i] = vecmat[i]
    end
    ret
end


if length(ARGS) != 2
    println("usage: gen_priors.jl [catalog.fits] [out_file.dat]")
else
    c0, r0 = read_r_colors(ARGS[1])
    if false && contains(ARGS[1], "stars")
        c0_quasar, r0_quasar = read_quasar_catalog("xdcore_005972.fits")
        c0 = vcat(c0, c0_quasar)
        r0 = vcat(r0, r0_quasar)
    end

    fit_r = fit_mle(LogNormal, r0)
    println(fit_r)

    NUM_COLOR_COMPONENTS = 8
    c0_train = c0
    #c0_test = c0[120001:end, :]
    fit_gmm = GMM(NUM_COLOR_COMPONENTS, c0_train, kind=:full, method=:split)
    println("train avll:", GaussianMixtures.avll(fit_gmm, c0_train))
    #println("test avll:", GaussianMixtures.avll(fit_gmm, c0_test))

    save(ARGS[2], "r_params", collect(params(fit_r)),
        "c_weights", weights(fit_gmm),
        "color_means", means(fit_gmm)',
        "color_covs", vecmat_to_tensor(covars(fit_gmm)))
end
