module DerivativeTestUtils

using Celeste: Model, DeterministicVI, SensitiveFloats
using Base.Test
import ForwardDiff

export forward_diff_model_params,
       unwrap_vp_vector,
       wrap_vp_vector,
       test_with_autodiff


""""
Return an ElboArgs with the corresponding type.
"""
function forward_diff_model_params{T<:Number}(::Type{T}, ea0::ElboArgs{Float64})
    P = length(ea0.vp[1])
    vp = Vector{T}[zeros(T, P) for s=1:ea0.S]
    # Set the values (but not gradient numbers) for parameters other
    # than the galaxy parameters.
    for s=1:ea0.S, i=1:length(ids)
        vp[s][i] = ea0.vp[s][i]
    end

    ElboArgs(ea0.images, vp, ea0.patches, ea0.active_sources)
end


"""
Wrap a vector of canonical parameters for active sources into a ElboArgs
object of the appropriate type. Used for testing with forward
autodifferentiation.
"""
function unwrap_vp_vector{NumType <: Number}(
            vp_vec::Vector{NumType}, ea::ElboArgs)
    vp_array = reshape(vp_vec, length(CanonicalParams), length(ea.active_sources))
    ea_local = forward_diff_model_params(NumType, ea)
    for sa = 1:length(ea.active_sources)
        ea_local.vp[ea.active_sources[sa]] = vp_array[:, sa]
    end
    ea_local
end


"""
Convert the variational params into a vector for autodiff.
"""
function wrap_vp_vector(ea::ElboArgs, use_active_sources::Bool)
    P = length(CanonicalParams)
    S = use_active_sources ? length(ea.active_sources) : ea.S
    x_mat = zeros(Float64, P, S)
    for s in 1:S
        ind = use_active_sources ? ea.active_sources[s] : s
        x_mat[:, s] = ea.vp[ind]
    end
    x_mat[:]
end


"""
Use ForwardDiff to test that fun(x) = sf (to abuse some notation)
"""
function test_with_autodiff{F}(fun::F, x::Vector{Float64}, sf::SensitiveFloat)

    const test_grad = false
    const test_hess = false

    @test fun(x) ≈ sf.v[]

    if test_grad
        ad_grad = ForwardDiff.gradient(fun, x)
        @test ad_grad ≈ sf.d[:]
    end

    if test_hess
        ad_hess = ForwardDiff.hessian(fun, x)
        @test ad_hess ≈ sf.h
    end
end


end # end module
