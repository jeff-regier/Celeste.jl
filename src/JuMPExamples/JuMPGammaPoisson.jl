using JuMP
using Distributions
using Gadfly



N = 5000
S = 5
point_spread = (0.5 / S) / 5

#################################
# Generate the celestial objects
mu = linspace(0.25, 0.75, S)

star_prob = 0.5
prior_upsilon = [1.0, 10.0]
prior_phi = [0.5, 0.5]

star_brightness = rand(Gamma(prior_upsilon[1], prior_phi[1]), S)
galaxy_brightness = rand(Gamma(prior_upsilon[2], prior_phi[2]), S)
true_is_star = rand(Bernoulli(star_prob), S)
true_brightness = true_is_star .* star_brightness + (1 - true_is_star) .* galaxy_brightness

# Generate the readings
m_loc = linspace(0, 1, N)
# Why is this convert statement necessary?
phi_ns = convert(Array{Float64, 2}, [pdf(Normal(0, point_spread), m_i - mu_s) for m_i in m_loc, mu_s in mu])
x = [convert(Float64, rand(Poisson(b))) for b in phi_ns * true_brightness]

plot(x=m_loc, y=x)

#########################
# Optimize with JuMP

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
@defNLExpr(e_log_fn[n=1:N], log(e_fn[n]) - var_fn[n] / (2 * e_fn[n] ^ 2))
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


# Define the model using the above expressions.
m = Model()
log_gamma_start = repmat(log(prior_upsilon)', S)
log_zeta_start = repmat(log(prior_phi)', S)
@defVar(m, log_gamma[s=1:S, a=1:2], start=log_gamma_start[s, a])
@defVar(m, log_zeta[s=1:S, a=1:2], start=log_zeta_start[s, a])
@defVar(m, logit_chi[s=1:S], start=[0 for s=1:S][s])
@setNLObjective(m, Max, e_log_lik + entropy + priors)
solve(m)


# Check the model output.
m_log_gamma = getValue(log_gamma)
m_log_zeta = getValue(log_zeta)
m_logit_chi = getValue(logit_chi)

m_chi = [ exp(m_logit_chi[s]) / (1 + exp(m_logit_chi[s])) for s in 1:S]
m_brightness = [ exp(m_log_gamma[s, a]) * exp(m_log_zeta[s, a]) for s in 1:S, a in 1:2]

m_total_brightness = [ m_chi[s] * m_brightness[s, 1] + (1 - m_chi[s]) * m_brightness[s, 2] for s in 1:S]


######
# Just a hacky way to get the value of an NLExpr.

debug_model = Model()
epsilon = 1e-6
@defVar(debug_model, 5 <= debug_var <= 5 + epsilon)
@defNLExpr(log_debug_var, log(debug_var))
@setNLObjective(debug_model, Max, log_debug_var)
solve(debug_model)
getObjectiveValue(debug_model)

########

using JuMP
using ReverseDiffSparse
m = Model()
@defVar(m, 0 <= beta <= 1)
@defExpr(beta2, beta * beta)
setValue(beta, 2.0)
getValue(beta)
getValue(beta2)
@defNLExpr(beta2nl[s=1:5], s * beta * beta)
[ ReverseDiffSparse.getvalue(beta2nl[s], m.colVal) for s=1:5 ]


########

m = Model()
@defVar(m, baz)
@defNLExpr(foo[1], 5 * baz)
@defNLExpr(foo[2], -3 * baz)
@defNLExpr(bar, sum{foo[i], i=1:2})
setValue(baz, 2)
ReverseDiffSparse.getvalue(bar, m.colVal) # Expect 4, get -12

############

m = Model()
@defVar(m, baz)
@defExpr(foo[1], 5 * baz)
@defExpr(foo[2], -3 * baz)

########

m = Model()
@defVar(m, baz)
@defNLExpr(foo[s=1:2], (s == 1) * 5 * baz + (s == 2) * (-3 * baz))
@defNLExpr(bar, sum{foo[i], i=1:2})
setValue(baz, 2)
ReverseDiffSparse.getvalue(bar, m.colVal) # Get 4



##############
# Arrays of expressions

m = Model()
@defVar(m, bar)

foo = Array(Any, 5, 10)
for i = 1:5
	for j = 1:10
		foo[i, j] = @defNLExpr(placeholder, i * bar + j)
	end
end
setValue(bar, 2)

# This works:
foo_mat = [ ReverseDiffSparse.getvalue(foo[i, j], m.colVal) for i=1:5, j=1:10]

# The result of this is a parametric expression, not a value:
@defNLExpr(foo_element, foo[1, 1])
ReverseDiffSparse.getvalue(foo_element, m.colVal)

# The result of this is an error:
@defNLExpr(foo_sum, sum{foo[i, j], i=1:5, j=1:10})
ReverseDiffSparse.getvalue(foo_sum, m.colVal)
# ERROR: `+` has no method matching +(::Float64, ::ParametricExpression{0})
#  in _EXPRVAL_ at /home/rgiordan/.julia/v0.3/ReverseDiffSparse/src/revmode.jl:678
#  in getvalue at /home/rgiordan/.julia/v0.3/ReverseDiffSparse/src/revmode.jl:696


##############################
# Structures of expressions?  No.

m = Model()
@defVar(m, baz)
immutable MyThingy
	foo::Matrix{Any}
	bar::Any
	MyThingy(offset::Float64) = begin
		foo = Array(Any, 5, 10)
		for i = 1:5
			for j = 1:10
				foo[i, j] = @defNLExpr(placeholder, i * baz + j + offset)
			end
		end
		bar = @defNLExpr(baz ^ 2 + offset)
		new(foo, bar)
	end
end

thingy = MyThingy(-4.0)

@defNLExpr(thingy_sum, sum{thingy.foo[i, j], i=1:5, j=1:10});
setValue(baz, 2)
ReverseDiffSparse.getvalue(thingy_sum, m.colVal)



######################
m = Model()
@defVar(m, foo)
setValue(foo, 2)
@defNLExpr(bar[ink=1:2, indigo=1:2, igloo=1:2, icarus=1:2],
	       foo * ink * indigo * igloo * icarus)
[ ReverseDiffSparse.getvalue(bar[i, j, k, l], m.colVal) for i=1:2, j=1:2, k=1:2, l=1:2 ]
