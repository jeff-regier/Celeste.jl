# Profilling JuMP code
# http://docs.julialang.org/en/release-0.1/stdlib/profile/

using JuMP
using Gadfly

function OptimizeExponentialModel(x::Array{Float64, 1})
	m = Model()
	n = length(x)
	@defVar(m, -1 <= alpha <= 1)
	@defNLExpr(exp_alpha_x[i=1:n], exp(x[i] * alpha))
	@setNLObjective(m, Min, sum{exp_alpha_x[i], i=1:n})
	solve(m)
	println("alpha = ", getValue(alpha))
	return getObjectiveValue(m)
end

function OptimizeExponentialModelExtraExpr(x::Array{Float64, 1})
	m = Model()
	n = length(x)
	@defVar(m, -1 <= alpha <= 1)
	@defNLExpr(alpha_x[i=1:n], x[i] * alpha)
	@defNLExpr(exp_alpha_x[i=1:n], exp(alpha_x[i]))
	@setNLObjective(m, Min, sum{exp_alpha_x[i], i=1:n})
	solve(m)
	println("alpha = ", getValue(alpha))
	return getObjectiveValue(m)
end


function EvaluateModelSize(n::Int64, ModelFunction::Function)
	# Note that @allocated doesn't actually assign variables, so I'll assigne
	# x twice, once to check its memory and once to actually assign the model
	# data.  (The allocated size scales nearly but not exactly linearly
    # with n.)
	x_size = @allocated x = randn(n)
	x = randn(n)
	model_size = @allocated ModelFunction(x)
	x_size, model_size
end

# Run once to compile so the JIT compiling isn't counted.
EvaluateModelSize(100, OptimizeExponentialModel)
EvaluateModelSize(100, OptimizeExponentialModelExtraExpr)

max_exponent = 6
x_sizes = zeros(max_exponent)
x_sizes_extra = zeros(max_exponent)

model_sizes = zeros(max_exponent)
model_sizes_extra = zeros(max_exponent)

for i = 1:max_exponent
	data_size = 5 * (10^i)
    print("Running for size ", i, "\n")
    x_sizes[i], model_sizes[i] = EvaluateModelSize(data_size, OptimizeExponentialModel)
    x_sizes_extra[i], model_sizes_extra[i] = EvaluateModelSize(data_size, OptimizeExponentialModelExtraExpr)
end



plot(layer(x=x=log10(x_sizes), y=log10(model_sizes), Geom.line, Geom.point),
  	 layer(x=x=log10(x_sizes_extra), y=log10(model_sizes_extra), Geom.line, Geom.point))
