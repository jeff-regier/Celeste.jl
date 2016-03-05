module Util


export matvec222, logit, inv_logit

function matvec222(mat::Matrix, vec::Vector)
    # x' A x in a slightly more efficient form.
    (mat[1,1] * vec[1] + mat[1,2] * vec[2]) * vec[1] +
            (mat[2,1] * vec[1] + mat[2,2] * vec[2]) * vec[2]
end

"""
Unpack a rotation-parameterized BVN covariance matrix.

Args:
  - ab: The ratio of the minor to the major axis.
  - angle: Rotation angle (in radians)
  - scale: The major axis.

 Returns:
   The 2x2 covariance matrix parameterized by the inputs.
"""
function get_bvn_cov{NumType <: Number}(ab::NumType, angle::NumType, scale::NumType)

    #@assert -pi/2 <= angle < pi/2
    if NumType <: AbstractFloat
        @assert 0 < scale
        @assert 0 < ab <= 1.
    end
    cp, sp = cos(angle), sin(angle)
    R = [[cp -sp]; [sp cp]]  # rotates
    D = diagm([1., ab])  # shrinks the minor axis
    W = scale * D * R'
    W' * W  # XiXi
end


end
