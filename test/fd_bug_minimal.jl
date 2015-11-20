using ForwardDiff

function gradind(x)
  find(ForwardDiff.grad(x))
end

# The product must be greater than 10 for gradind() to work.
H = 3
W = 4

# Doesn't work -- it fills with pointers that are all the same!
x_mat = fill(zeros(Float64, W), H)
x_mat[1][1] = 5.0
println(x_mat)

x_mat = Array(Array{Float64}, H)
for h=1:H
  x_mat[h] = zeros(Float64, W)
end
x_mat[1][1] = 5.0
println(x_mat)

x_vec = rand(H * W)

# Use these global variables to read the ForwardDiff types and variables
# out of the function.
FDType = 0.0
x_vec_fd = 0.0

function fun2{T <: Number}(x_vec::Vector{T})
  # Read into global variables for further debugging.
  global FDType, x_vec_fd
  FDType = T
  x_vec_fd = deepcopy(x_vec)

  # Note: this is failing even for Float64.
  # Populate a matrix from x_vec.
  x_mat = fill(zeros(FDType, W), H);
  for h=1:H, w in 1:W
    index = w + W  * (h - 1)
    println(index)
    x_mat[h][w] = x_vec_fd[index]
  end

  tot = zero(T)
  for h=1:H
    tot += sum(x_mat[h])
  end

  tot
end

fun2(x_vec)
g = ForwardDiff.gradient(fun2)
g(x_vec)

########################
# Fails:
x_mat = fill(zeros(FDType, W), H);
for h=1:H, w in 1:W
  x_mat[h][w] = x_vec_fd[w + W  * (h - 1)]
end
for h=1:H
  println([ gradind(x_mat[h][w]) for w=1:W ])
end


# Succeeds:
x_mat = fill(zeros(FDType, W), H);
for h=1:H
  x_mat[h] = x_vec_fd[(1:W) + W  * (h - 1)]
end
for h=1:H
  println([ gradind(x_mat[h][w]) for w=1:W ])
end
