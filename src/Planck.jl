# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module Planck

export photons_expected

using DualNumbers
using CelesteTypes



# constants

const h = 6.6260693e-34 # Planck constant (Joules * second)
const k = 1.3806488e-23 # Boltzmann constant (Joules / Kelvin)
const c = 299792458 # speed of light (meters / second)

const hc_k = (h * c) / k
const hpicc2 = 2 * pi * h * c * c

#Stefan-Boltzmann constant
const sigma = 2 * pi^5 * k^4 / (15 * c^2 * h^3) # J / (s * m^2 * K^4)

const sun_wattage = 3.839 * 1e26
const sun_radius = 6.995e8
const m_per_ly = c * 31556952.

const lens_area = .75 * pi * 1.25^2 # in meters^2
const exposure_duration = 54.

const bands = ['z', 'i', 'r', 'g', 'u']

const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

# load the filter curves

function load_filter_curves()
	filter_curves = open("$dat_dir/filter_curves")

	wavelength_lookup = Array(Array{Float64, 1}, 5)
	sensitivity_lookup = Array(Array{Float64 , 1}, 5)
	for i in 1:5
		wavelength_lookup[i] = Array(Float64, 0)
		sensitivity_lookup[i] = Array(Float64, 0)
	end

	for line in eachline(filter_curves)
		band, wavelength, sensitivity = split(strip(line), "\t")
		band_id = findin(bands, band[1])[]
		push!(wavelength_lookup[band_id], float(wavelength) * 1e-4 * 1e-6) #aenstroms to meters
		push!(sensitivity_lookup[band_id], float(sensitivity))
	end

	close(filter_curves)

	return wavelength_lookup, sensitivity_lookup
end


wavelength_lookup, sensitivity_lookup = load_filter_curves()


# compute the expected number of photons

function photons_per_joule(T, band_id::Int64)
	x = wavelength_lookup[band_id] 
	radiances = hpicc2 ./ (x.^5 .* (exp(hc_k ./ (x .* T)) .- 1))
	total_radiance = sigma * T^4 #across bands, per m^2
	radiance_densities = radiances ./ total_radiance

	photon_energies = (h * c ./ x) # Joules
	photon_fluxes = radiance_densities ./ photon_energies
	filtered_photon_fluxes = photon_fluxes .* sensitivity_lookup[band_id]

	avg_photons = mean(filtered_photon_fluxes) #per hertz
	range = abs(x[2] - x[1]) * length(x)
	return avg_photons * range #approximates the integral
end

function photons_expected(T, solar_L::Float64, d::Float64, band_id::Int64)
	L = solar_L * sun_wattage
	D = d * m_per_ly
	lens_prop = lens_area ./ (4pi * D^2)
	lens_watts = lens_prop * L
	return photons_per_joule(T, band_id) * lens_watts * exposure_duration
end

function photons_expected(T, solar_L::Float64, d::Float64)
	return Dual{Float64}[photons_expected(T, solar_L, d, b) for b in 1:5]
end


#= for testing

# sensitivity_lookup[1] = [1. for i in 1:1000000]
# wavelength_lookup[1] = [1:1000000] * 1e-10

# the sun
# T = 6000.
# d_in_m = 150e9
# sun_radius^2 * 4pi
# photons_per_joule(T, 1) * sun_wattage
# photons_expected(T, 1., d_in_m / m_per_ly) # 5e22 photons in the r band?

# vega
# surface_area(9602., 40.12 * sun_wattage)
# pi * (196e7)^2
# photons_expected(9602., 40.12, 25.04)  ### 6e11 photons in the r band?

# arcturus
# photons_expected(4290., 170., 36.7)  ### 1e12 photons in the r band?

# a supergiant on the edge of the milky way?
# photons_expected(3602., 400000., 90e3) ### 4e8 photons in the r band?

# our sun at the edge of the milky way?
=# photons_expected(6000., 1., 90e3) ### 1685 photons in the r band?

function make_tau(v::Float64, dv::Float64)
	make_singleton_param(:tau, 1, v, dv)
end

function expected_colors(tau::SourceParam)
	dual_tau = dual(tau.v, 1.)
    planck_denorm = photons_expected(dual_tau, 1., 1.)
    planck_colors = log(planck_denorm[2:5]) - log(planck_denorm[1:4])
	[make_tau(real(color), epsilon(color)) for color in planck_colors]
end


end



