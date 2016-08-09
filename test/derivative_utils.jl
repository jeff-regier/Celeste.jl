import Base.convert

# function convert(FDType::Type{ForwardDiff.GradientNumber},
#                  ea::ElboArgs{Float64})
#     x = ea.vp[1]
#     P = length(x)
#     FDType = ForwardDiff.GradientNumber{length(ea.vp[1]), Float64}
#
#     fd_x = [ ForwardDiff.GradientNumber(x[i], zeros(Float64, P)...) for i=1:P ]
#     convert(FDType, x[1])
#
#     vp_fd = convert(Array{Array{FDType, 1}, 1}, ea.vp[1])
#     ea_fd = ElboArgs(vp_fd)
# end
#
# function convert(FDType::Type{ForwardDiff.HessianNumber},
#                  ea::ElboArgs{Float64})
#     x = ea.vp[1]
#     P = length(x)
#     FDType = ForwardDiff.HessianNumber{length(ea.vp[1]), Float64}
#
#     fd_x = [ ForwardDiff.HessianNumber(x[i], zeros(Float64, P)...) for i=1:P ]
#     convert(FDType, x[1])
#
#     vp_fd = convert(Array{Array{FDType, 1}, 1}, ea.vp[1])
#     ea_fd = ElboArgs(vp_fd)
# end


# Maybe write it as a convert()?
function forward_diff_model_params{T<:Number}(::Type{T}, ea0::ElboArgs{Float64})
    P = length(ea0.vp[1])
    vp = Vector{T}[zeros(T, P) for s=1:ea0.S]
    # Set the values (but not gradient numbers) for parameters other
    # than the galaxy parameters.
    for s=1:ea0.S, i=1:length(ids)
        vp[s][i] = ea0.vp[s][i]
    end

    ElboArgs(ea0.images,
             vp,
             ea0.tile_source_map,
             ea0.patches,
             ea0.active_sources)
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
function test_with_autodiff(fun::Function, x::Vector{Float64}, sf::SensitiveFloat)
    ad_grad = ForwardDiff.gradient(fun, x)
    ad_hess = ForwardDiff.hessian(fun, x)
    @test_approx_eq fun(x) sf.v
    @test_approx_eq ad_grad sf.d[:]
    @test_approx_eq ad_hess sf.h
end


"""
Set all but a few pixels to NaN to speed up autodiff Hessian testing.
"""
function trim_tiles!(tiled_blob::Vector{TiledImage}, keep_pixels)
    for b = 1:length(tiled_blob)
	    pixels1 = tiled_blob[b].tiles[1,1].pixels
        h_width, w_width = size(pixels1)
	    pixels1[setdiff(1:h_width, keep_pixels), :] = NaN
        pixels1[:, setdiff(1:w_width, keep_pixels)] = NaN
	end
end
