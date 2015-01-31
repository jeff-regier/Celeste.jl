#!/usr/bin/env julia

using FITSIO
using GaussianMixtures
using Distributions


function mag_to_nanomaggies(mag::Float64)
	10 ^ (( mag - 22.5 ) / -2.5)
end

@vectorize_1arg Float64 mag_to_nanomaggies


function read_r_colors(prior_file)
	cat = fits_open_table(ENV["DAT"]"/priors/"prior_file)
	num_rows = int(fits_read_keyword(cat, "NAXIS2")[1])
	table = Array(Float64, num_rows, 12)
	for i in 1:12
		data = Array(Float64, num_rows)
		fits_read_col(cat, Float64, i, 1, 1, data)
		table[:, i] = data
	end
	fits_close_file(cat)

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


c0, r0 = read_r_colors("stars.fits")
c1, r1 = read_r_colors("gals.fits")

star_r = fit_mle(Gamma, r0)
galaxy_r = fit_mle(Gamma, r1)


r_file = open(ENV["DAT"]"/r_prior.dat", "w+")
serialize(r_file, (params(star_r), params(galaxy_r)))
close(r_file)


D = length(ARGS) != 0 ? int(ARGS[1]) : 8
c0_train = c0
#c0_test = c0[120001:end, :]
gmm_star = GMM(D, c0_train, kind=:full, method=:split)
println("star train avll:", GaussianMixtures.avll(gmm_star, c0_train))
#println("star test avll:", GaussianMixtures.avll(gmm_star, c0_test))

c1_train = c1
#c1_test = c1[120001:end, :]
gmm_gal = GMM(D, c1_train, kind=:full, method=:split)
println("gal train avll:", GaussianMixtures.avll(gmm_gal, c1_train))
#println("gal test avll:", GaussianMixtures.avll(gmm_gal, c1_test))


c_file = open(ENV["DAT"]"/ck_prior.dat", "w+")
serialize(c_file, ((weights(gmm_star), means(gmm_star)', covars(gmm_star)),
	(weights(gmm_gal), means(gmm_gal)', covars(gmm_gal))))
close(c_file)


