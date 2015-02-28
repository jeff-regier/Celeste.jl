using JuMP
using Gadfly
using Distributions

N = 1000
S = 2

# Generate the celestial objects
mu = linspace(0.25, 0.75, S)

star_prob = 0.5
star_brightness_mean = 0.0
star_brightness_sd = 0.6
galaxy_brightness_mean = 2.3
galaxy_brightness_sd = 0.6

star_brightness = rand(LogNormal(star_brightness_mean, star_brightness_sd), S)
galaxy_brightness = rand(LogNormal(galaxy_brightness_mean, galaxy_brightness_sd), S)
true_is_star = rand(Bernoulli(star_prob), S)
true_brightness = true_is_star .* star_brightness + (1 - true_is_star) .* galaxy_brightness

# Generate the readings
m = linspace(0, 1, N)
# Why is this convert statement necessary?
phi_ns = [convert(Float64, pdf(Normal(0, 0.06), m_i - mu_s)) for m_i in m, mu_s in mu]
x = [rand(Poisson(b)) for b in phi_ns * true_brightness]



