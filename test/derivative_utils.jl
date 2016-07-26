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
