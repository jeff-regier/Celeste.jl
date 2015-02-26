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
@setObjective(m, Min, sum(xb))
status = solve(m)
getObjectiveValue(m)




n = 100
k = 3
x = randn(n, k)
true_beta = randn(k)
y = x * true_beta + randn(n)


m = Model(solver=NLoptSolver(algorithm=:LD_LBFGS))
@defVar(m, -10 <= gamma <= 10)
index = 2
@defNLExpr(eps[i=1:n], gamma)
@defNLExpr(eps_sum, sum{eps})
@setNLObjective(m, Min, eps_sum)
status = solve(m)
getObjectiveValue(m)


@setNLObjective(m, Min, sum{eps, i=1:n})

@defVar(m, -10 <= beta[1:k] <= 10)

cost = [1, 2, 3]
index = 2

#@defNLExpr(error2[i=1:k], beta[i])

#for this_row in 1:n 
#  @defNLExpr(error2[i=this_row], (y - beta[1] * x[this_row, 1] - beta[2] * x[this_row, 2] - beta[3] * x[this_row, 3])^2)
#end
#@defNLExpr(error2[1], (y - beta[1] * x[1, 1] - beta[2] * x[1, 2] - beta[3] * x[1, 3])^2)



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

