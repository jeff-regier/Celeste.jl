# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module Util

VERSION < v"0.4.0-dev" && using Docile

export matvec222, logit, inv_logit

function matvec222(mat::Matrix, vec::Vector)
    # x' A x in a slightly more efficient form.
    (mat[1,1] * vec[1] + mat[1,2] * vec[2]) * vec[1] +
            (mat[2,1] * vec[1] + mat[2,2] * vec[2]) * vec[2]
end

function get_bvn_cov{NumType <: Number}(ab::NumType, angle::NumType, scale::NumType)
    # Unpack a rotation-parameterized BVN covariance matrix.
    #
    # Args:
    #   - ab: The ratio of the minor to the major axis.
    #   - angle: Rotation angle (in radians)
    #   - scale: The major axis.
    #
    #  Returns:
    #    The 2x2 covariance matrix parameterized by the inputs.

    #@assert -pi/2 <= angle < pi/2
    @assert 0 < scale
    @assert 0 < ab <= 1.
    cp, sp = cos(angle), sin(angle)
    R = [[cp -sp], [sp cp]]  # rotates
    D = diagm([1., ab])  # shrinks the minor axis
    W = scale * D * R'
    W' * W  # XiXi
end

function inv_logit{NumType <: Number}(x::NumType)
    # TODO: bounds checking
    -log(1.0 ./ x - 1)
end

function logit{NumType <: Number}(x::NumType)
    # TODO: bounds checking
    1.0 ./ (1.0 + exp(-x))
end

end