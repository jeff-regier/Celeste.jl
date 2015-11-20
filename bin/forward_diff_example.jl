#############################
# Try to understand ForwardDiff.

using ForwardDiff

xd = 0

function f2{T <: Number}(x::Vector{T})
  global xd
  println(typeof(x))
  println(size(x))
  xT = typeof(x[1])
  println(xT)
  println(isa(xT, Number))
  value = sum(x .* x)
  xd = x
  println("value type:")
  println(typeof(value))
  value
end

g = ForwardDiff.gradient(f2)
h = ForwardDiff.hessian(f2)

x = rand(length(ids));
FDType = typeof(fd[1])

########


using Celeste
using CelesteTypes
using Base.Test
using Distributions
using SampleData
using Transform
using PyPlot

using ForwardDiff

import Synthetic
import WCS

println("Running hessian tests.")

function gradind(x)
  find(ForwardDiff.grad(x))
end

blob, mp, three_bodies, tiled_blob = gen_three_body_dataset();
kept_ids = 1:length(ids);
omitted_ids = setdiff(1:length(ids), kept_ids);

mp_fd = 0.0;
x_fd_vec = 0.0;
x_fd = 0.0;

transform = Transform.get_identity_transform(length(ids), mp.S);
function example{T <: Number}(x::Vector{T})
  global mp_fd
  global x_fd
  global x_fd_vec
  mp_fd = CelesteTypes.forward_diff_model_params(T, mp);
  x_mat = reshape(x, length(kept_ids), mp.S)
  x_fd = deepcopy(x_mat)
  x_fd_vec = deepcopy(x)
  transform.array_to_vp!(x_mat, mp_fd.vp, omitted_ids);


  tot = zero(T)
  for s = 1:mp_fd.S, id in kept_ids
    tot += mp_fd.vp[s][id]
  end
  tot
end


g = ForwardDiff.gradient(example)
x = transform.vp_to_array(mp.vp, omitted_ids);
grad = g(x[:])
FDType = typeof(x_fd[1,1])
reshape(grad, length(kept_ids), mp.S)


######################################
#

include("src/ElboNoDeriv.jl")
blob, mp, three_bodies, tiled_blob = gen_sample_star_dataset();
kept_ids = 1:length(ids);
omitted_ids = setdiff(1:length(ids), kept_ids);

sb = ElboNoDeriv.SourceBrightness(mp_fd.vp[1]);
@time elbo = ElboNoDeriv.elbo(tiled_blob, mp_fd);
@time elbo = ElboDeriv.elbo(tiled_blob, mp);

@time foo = mp_fd.vp

@time for n=1:10000
  foo = sum([ mp_fd.vp[1][i] * mp_fd.vp[1][i] for i=1:length(ids)] )
end

@time for n=1:10000
  bar = sum([ mp.vp[1][i] * mp.vp[1][i] for i=1:length(ids)] )
end
