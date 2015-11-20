using ForwardDiff

function gradind(x)
  find(ForwardDiff.grad(x))
end

H = 4
W = 3

x_vec = rand(H * W)

function fun{T <: Number}(x_vec::Vector{T})
  x_mat = reshape(x_vec, H, W)
  sum(x_mat)
end

g = ForwardDiff.gradient(fun)
g(x_vec)


FDType = 0.0
x_vec_fd = 0.0

function fun2{T <: Number}(x_vec::Vector{T})
  global FDType, x_vec_fd
  FDType = T
  x_vec_fd = deepcopy(x_vec)

  x_mat = [zeros(T, W) for h=1:H]
  tot = zero(T)
  for h=1:H
    x_mat[h] = x_vec[(1:W) + W  * (h - 1)]
  end

  for h=1:H
    tot += sum(x_mat[h])
  end

  tot
end

g = ForwardDiff.gradient(fun2)
g(x_vec)

x_mat = [zeros(FDType, W) for h=1:H];
for h=1:H
  x_mat[h] = x_vec_fd[(1:W) + W  * (h - 1)]
  println([ gradind(x_mat[h][w]) for w=1:W ])
end
