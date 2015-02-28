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
x = [convert(Float64, rand(Poisson(b))) for b in phi_ns * true_brightness]

#plot(x=m,y=x)



# Optimize with JuMP
m = Model()
@defVar(m, 0.001 <= gamma[1:S, 1:2] <= 10)
@defVar(m, 0.001 <= zeta[1:S, 1:2] <= 10)
@defVar(m, 0.001 <= chi[1:S] <= 1 - 0.001)

# Set values to help with debugging?  It would be nice if we could
# evaluate expressions one by one.
for s in 1:S
	setValue(gamma[s, 1], 5.0)
	setValue(zeta[s, 1], 5.0)
	setValue(gamma[s, 2], 3.0)
	setValue(zeta[s, 2], 3.0)
    setValue(chi[s], 0.4)
end

# Define the entropy.
@defNLExpr(ent_rsa[s=1:S, a=1:2],
	       gamma[s, a] + log(zeta[s, a]) + lgamma(zeta[s, a]))
@defNLExpr(ent_as[s=1:S],
	       -1 * chi[s] * log(chi[s]) - (1 - chi[s]) * log(1 - chi[s]))
@defNLExpr(entropy, sum{ent_rsa[s, a], s=1:S, a=1:2} + sum{ent_as[s], s=1:S})
@setNLObjective(m, Max, entropy)

# This gives 
# ERROR: InexactError()
# in float64 at float.jl:55
solve(m)


# Hmm, maybe lgamma doesn't work.  Nope:
m = Model()
@defVar(m, 1 <= x <= 2)
@setNLObjective(m, Max, lgamma(x))
solve(m)



# The rest of the model:

# Optimize with JuMP
m = Model()
@defVar(m, 0.001 <= gamma[1:S, 1:2] <= 10)
@defVar(m, 0.001 <= zeta[1:S, 1:2] <= 10)
@defVar(m, 0.001 <= chi[1:S] <= 1 - 0.001)

# Define the r expectations.
@defNLExpr(e_ra[s=1:S, a=1:2], gamma[s, a] * zeta[s, a])
@defNLExpr(e_ra2[s=1:S, a=1:2], (1 + gamma[s, a]) * gamma[s, a] * (zeta[s, a] ^ 2))
@defNLExpr(e_r[s=1:S], chi[s] * e_ra[s, 1] + (1 - chi[s]) * e_ra[s, 2])
@defNLExpr(var_r[s=1:S], chi[s] * e_ra2[s, 1] + (1 - chi[s]) * e_ra2[s, 2] - (e_r[s]) ^ 2)

# Define the F expectations.
@defNLExpr(e_fns[n=1:N, s=1:S], e_r[s] * phi_ns[n, s])
@defNLExpr(var_fns[n=1:N, s=1:S], var_r[s] * (phi_ns[n, s])^2)
@defNLExpr(e_fn[n=1:N], sum{e_fns[n, s], s=1:S})
@defNLExpr(var_fn[n=1:N], sum{var_fns[n, s], s=1:S})

# This is the problem:
@defNLExpr(e_log_fn[n=1:N], 2 * log(e_fn[n]) - var_fn[n] / (2 * e_fn[n] ^ 2))
@defNLExpr(e_log_lik, sum{x[n] * e_log_fn[n] - e_fn[n], n=1:N})

@setNLObjective(m, Max, e_log_lik)
solve(m)