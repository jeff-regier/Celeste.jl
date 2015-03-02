using JuMP
using Distributions
using Gadfly

N = 500
S = 3

#################################
# Generate the celestial objects
mu = linspace(0.25, 0.75, S)

star_prob = 0.5
prior_upsilon = [1.0, 4.0]
prior_phi = [0.5, 0.5]

star_brightness = rand(Gamma(prior_upsilon[1], prior_phi[1]), S)
galaxy_brightness = rand(Gamma(prior_upsilon[2], prior_phi[2]), S)
true_is_star = rand(Bernoulli(star_prob), S)
true_brightness = true_is_star .* star_brightness + (1 - true_is_star) .* galaxy_brightness

# Generate the readings
m_loc = linspace(0, 1, N)
# Why is this convert statement necessary?
phi_ns = convert(Array{Float64, 2}, [pdf(Normal(0, 0.06), m_i - mu_s) for m_i in m_loc, mu_s in mu])
x = [convert(Float64, rand(Poisson(b))) for b in phi_ns * true_brightness]

plot(x=m_loc, y=x)

#########################
# Optimize with JuMP
m = Model()
@defVar(m, log_gamma[1:S, 1:2], start=log(true_brightness[s]))
@defVar(m, log_zeta[1:S, 1:2], start=)
@defVar(m, logit_chi[1:S])

# Unconstrain the variables.  Does this really help at all?
@defNLExpr(q_chi[s=1:S], exp(logit_chi[s]) / (1 + exp(logit_chi[s])))
@defNLExpr(q_gamma[s=1:S, a=1:2], exp(log_gamma[s, a]))
@defNLExpr(q_zeta[s=1:S, a=1:2], exp(log_zeta[s, a]))

# Define the a expectations.
@defNLExpr(e_a[s=1:S, 1], q_chi[s])
@defNLExpr(e_a[s=1:S, 2], 1 - q_chi[s])

# Define the r expectations.
@defNLExpr(e_ra[s=1:S, a=1:2],     q_gamma[s, a] * q_zeta[s, a])
@defNLExpr(e_ra2[s=1:S, a=1:2],    (1 + q_gamma[s, a]) * q_gamma[s, a] * (q_zeta[s, a] ^ 2))
@defNLExpr(e_log_ra[s=1:S, a=1:2], digamma(q_gamma[s, a]) + log(q_zeta[s, a]))
@defNLExpr(e_r[s=1:S],             sum{e_a[s, a] * e_ra[s, a], a=1:2})
@defNLExpr(var_r[s=1:S],           sum{e_a[s, a] * e_ra2[s, a], a=1:2} - (e_r[s]) ^ 2)

# Define the F expectations.
@defNLExpr(e_fns[n=1:N, s=1:S], e_r[s] * phi_ns[n, s])
@defNLExpr(var_fns[n=1:N, s=1:S], var_r[s] * (phi_ns[n, s])^2)
@defNLExpr(e_fn[n=1:N], sum{e_fns[n, s], s=1:S})
@defNLExpr(var_fn[n=1:N], sum{var_fns[n, s], s=1:S})
@defNLExpr(e_log_fn[n=1:N], 2 * log(e_fn[n]) - var_fn[n] / (2 * e_fn[n] ^ 2))
@defNLExpr(e_log_lik, sum{x[n] * e_log_fn[n] - e_fn[n], n=1:N})

# Define the entropy.
@defNLExpr(ent_rsa[s=1:S, a=1:2],
	       q_gamma[s, a] + log(q_zeta[s, a]) + lgamma(q_zeta[s, a]) +
	       (1 - q_zeta[s, a]) * digamma(q_zeta[s, a]))
@defNLExpr(ent_as[s=1:S],
	       -1 * q_chi[s] * log(q_chi[s]) - (1 - q_chi[s]) * log(1 - q_chi[s]))
@defNLExpr(entropy, sum{ent_rsa[s, a], s=1:S, a=1:2} + sum{ent_as[s], s=1:S})

# Define the expected priors.
@defNLExpr(e_ra_prior[s=1:S, a=1:2],
	       (prior_upsilon[a] - 1) * e_log_ra[s, a] -
	       e_ra[s, a] / prior_phi[a] -
	       prior_upsilon[a] * log(prior_phi[a]) - lgamma(prior_upsilon[a]))
@defNLExpr(e_a_prior[s=1:S], q_chi[s] * log(star_prob) + (1 - q_chi[s]) * log(1 - star_prob))
@defNLExpr(priors, sum{e_a[s, a] * e_ra_prior[s, a], s=1:S, a=1:2} + sum{e_a_prior[s], s=1:S})

@setNLObjective(m, Max, e_log_lik + entropy + priors)
solve(m)

