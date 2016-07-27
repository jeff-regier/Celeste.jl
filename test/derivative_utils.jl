# Maybe write it as a convert()?
function forward_diff_model_params{T <: Number}(
            FDType::Type{T},
            ea0::ElboArgs{Float64})
    P = length(ea0.vp[1])
    vp = Vector{FDType}[zeros(FDType, P) for s=1:ea0.S]
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

