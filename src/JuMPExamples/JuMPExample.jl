using JuMP
using NLopt
using Clp
using Ipopt


# Memory usage.  I suspect that indexed JuMP objects are making a deep
# copy of the data that they use based on the following example.  I'm
# not sure how to check this more directly.
m = Model()
big_data = randn(100)
@defVar(m, big_parameter)
@defNLExpr(big_transform[i=1:100], big_parameter * big_data[i]^2)
big_data = -5
print(big_transform)
print(big_data)


################
# Exponential examples

# This example works.
n = 100
x = randn(n)
m = Model()
@defVar(m, -0.1 <= alpha <= 0.1)
@defNLExpr(eps[i=1:n], exp(x[i] * alpha)) # (1)
@defNLExpr(eps2[i=1:n], alpha * eps[i]) # (1)
@setNLObjective(m, Min, sum{eps2[i], i=1:n})
status = solve(m)
getObjectiveValue(m)
println("alpha = ", getValue(alpha))
sum(exp(getValue(alpha) * x))


# This example works with a vector inner product.
n = 100
x = randn(n, 2)
m = Model()
@defVar(m, -0.1 <= alpha[k in 1:2] <= 0.1)
@defVar(m, -0.1 <= gamma <= 0.1)
@defNLExpr(eps[i=1:n], exp(sum{x[i, k] * alpha[k], k=1:2}))  # works
#@defNLExpr(eps[i=1:n], exp(dot(vec(x[i, :]), alpha)))  # NB: does not work
@defNLExpr(eps2[i=1:n], gamma * eps[i])
@setNLObjective(m, Min, sum{eps2[i], i=1:n})
status = solve(m)
getObjectiveValue(m)
println("alpha = ", getValue(alpha))



###################
# OLS Examples

n = 100
k = 3
x = randn(n, k)
true_beta = randn(k)
y = x * true_beta + randn(n)

# This is stupid but is working with sums and array expressions.
#m = Model(solver=ClpSolver())
m = Model()
@defVar(m, -10 <= beta[1:k] <= 10)
# One row for example.
eps_aff = AffExpr(beta[j=1:k], vec(x[1,:]), -y[1])
@defExpr(eps[i=1:n], AffExpr(beta[j=1:k], vec(x[i,:]), -y[i]))
@setObjective(m, Min, sum{sum(eps), i=1:n})
status = solve(m)
getObjectiveValue(m)

# This example works for least squares.
m = Model()
@defVar(m, -10 <= beta[1:k] <= 10)
@defNLExpr(eps[i=1:n], y[i] - sum{x[i, j] * beta[j], j=1:k})
@defNLExpr(eps2[i=1:n], eps[i] * eps[i])
@defNLExpr(eps2_sum, sum{eps2[i], i=1:n}) # Note that only NLExprs can take NLExprs I think
@setNLObjective(m, Min, eps2_sum)
status = solve(m)
getObjectiveValue(m)
println("beta = ", getValue(beta))
print((x' * x)^(-1) * (x' * y))


# This does not work.
n = 100
k = 3
x = randn(n, k)
true_beta = randn(k)
y = x * true_beta + randn(n)

function CalculateEpsilon(index)
  @defNLExpr(eps_squared[index], (y[index] - sum{x[index, j] * beta[j], j=1:k})^2)
end
m = Model()
@defVar(m, -10 <= beta[1:k] <= 10)
for i = [n]
  CalculateEpsilon(i)
end

# Warning: at thie point, if eps_squared has ever been defined
# this may seem to work, but it is not actually being set in CalculateEpsilon.
@setNLObjective(m, Min, sum{eps_squared[i], i=1:n})
status = solve(m)
getObjectiveValue(m)
println("beta = ", getValue(beta))
print((x' * x)^(-1) * (x' * y))



#################
# A basic working example

function logit(x)
    exp(x) / (1 + exp(x))
end

m = Model(solver=NLoptSolver(algorithm=:LD_LBFGS))

@defVar(m, x)
@defVar(m, 0 <= y <= 30)

# This works:
@defNLExpr(aux, 2 * exp(x) / (1 + exp(x)))
@defNLExpr(sin_x, sin(x))

# This does not work:
#@defNLExpr(aux, 2 * logit(x))

# This works:
@defNLExpr(aux2, aux^2)


@setNLObjective(m, Max, 5aux2 + 3y + sin_x)
print(m)
status = solve(m)

println("Objective value: ", getObjectiveValue(m))
println("x = ", getValue(x))
println("y = ", getValue(y))



#################
# Here is a (bad) way to do it with nonlinear equality constraints.
# This is the only NLopt solver that accomodates nonlinear equality constriants.
m = Model(solver=NLoptSolver(algorithm=:GN_ISRES))

# Every variable must be bounded for GN_ISRES.
@defVar(m, -1e6 <= x <= 1e6)
@defVar(m, -10 <= aux <= 10)
@defVar(m, 0 <= y <= 30)

@addNLConstraint(m, aux == 2 * exp(x) / (1 + exp(x)))
@setNLObjective(m, Max, 5aux + 3y)
print(m)
status = solve(m)

println("Objective value: ", getObjectiveValue(m))
println("x = ", getValue(x))
println("aux = ", getValue(aux))
println("y = ", getValue(y))




###########
# An example from the JuMP web site:
N     = 1000
ni    = N
h     = 1/ni
alpha = 350

# So this doesn't work with :GN_ISRES.  I suspect problems fith NLopt.
#m = Model(solver=NLoptSolver(algorithm=:GN_ISRES))
m = Model()

@defVar(m, -1 <= t[1:(ni+1)] <= 1)
@defVar(m, -0.05 <= x[1:(ni+1)] <= 0.05)
@defVar(m, u[1:(ni+1)])

@setNLObjective(m, Min, sum{ 0.5*h*(u[i+1]^2 + u[i]^2) + 0.5*alpha*h*(cos(t[i+1]) + cos(t[i])), i = 1:ni})

# cons1
for i in 1:ni
    @addNLConstraint(m, x[i+1] - x[i] - (0.5h)*(sin(t[i+1])+sin(t[i])) == 0)
end
# cons2
for i in 1:ni
    @addConstraint(m, t[i+1] - t[i] - (0.5h)*u[i+1] - (0.5h)*u[i] == 0)
end

solve(m)

