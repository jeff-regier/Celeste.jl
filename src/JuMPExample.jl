using JuMP
using NLopt
using Clp


#################
# A stupid but more complicated example.



n = 100
k = 3
x = randn(n, k)
true_beta = randn(k)
y = x * true_beta + randn(n)

# This is stupid but is working with sums and array expressions.
m = Model(solver=ClpSolver())
@defVar(m, -10 <= beta[1:k] <= 10)
@defExpr(xb[i=1:n], dot(vec(x[i,:]), beta))
@defExpr(eps[i=1:n], y[i] - xb[i])
@setObjective(m, Min, sum(eps))
status = solve(m)
getObjectiveValue(m)

# Try a nonlinear version.  This does not work for reasons I don't understand.
m = Model(solver=NLoptSolver(algorithm=:LD_LBFGS))
@defVar(m, -10 <= beta[1:k] <= 10)
@defExpr(xb[i=1:n], dot(vec(x[i,:]), beta))
@defNLExpr(eps[i=1:n], (y[i] - xb[i])^2)
@setNLObjective(m, Min, sum{eps, i=1:n})
status = solve(m)
getObjectiveValue(m)





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

# So this doesn't work.  I suspect problems fith NLopt.
m = Model(solver=NLoptSolver(algorithm=:GN_ISRES))

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

