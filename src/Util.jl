# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module Util

export matvec222

function matvec222(mat::Matrix, vec::Vector)
    # x' A x in a slightly more efficient form.
    (mat[1,1] * vec[1] + mat[1,2] * vec[2]) * vec[1] +
            (mat[2,1] * vec[1] + mat[2,2] * vec[2]) * vec[2]
end

function get_bvn_cov(ab::Float64, angle::Float64, scale::Float64)
    # Unpack a rotation-parameterized BVN covariance matrix.
    #@assert -pi/2 <= angle < pi/2
    @assert 0 < scale
    @assert 0 < ab <= 1.
    cp, sp = cos(angle), sin(angle)
    R = [[cp -sp], [sp cp]]  # rotates
    D = diagm([1., ab])  # shrinks the minor axis
    W = scale * D * R'
    W' * W  # XiXi
end

end
