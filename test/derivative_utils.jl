import Base.convert


function convert(FDType::Type{ForwardDiff.GradientNumber},
                 ea::ElboArgs{Float64})
    x = ea.vp[1]
    P = length(x)
    FDType = ForwardDiff.GradientNumber{length(ea.vp[1]), Float64}

    fd_x = [ ForwardDiff.GradientNumber(x[i], zeros(Float64, P)...) for i=1:P ]
    convert(FDType, x[1])

    vp_fd = convert(Array{Array{FDType, 1}, 1}, ea.vp[1])
    ea_fd = ElboArgs(vp_fd)
end

function convert(FDType::Type{ForwardDiff.HessianNumber},
                 ea::ElboArgs{Float64})
    x = ea.vp[1]
    P = length(x)
    FDType = ForwardDiff.HessianNumber{length(ea.vp[1]), Float64}

    fd_x = [ ForwardDiff.HessianNumber(x[i], zeros(Float64, P)...) for i=1:P ]
    convert(FDType, x[1])

    vp_fd = convert(Array{Array{FDType, 1}, 1}, ea.vp[1])
    ea_fd = ElboArgs(vp_fd)
end


# TODO: Maybe write it as a convert()?
function forward_diff_model_params{T <: Number}(
            FDType::Type{T},
            base_ea::ElboArgs{Float64})
    S = length(base_mp.vp)
    P = length(base_mp.vp[1])
    ea_fd = ElboArgs{FDType}([zeros(FDType, P) for s=1:S]);
    # Set the values (but not gradient numbers) for parameters other
    # than the galaxy parameters.
    for s=1:base_mp.S, i=1:length(ids)
        ea_fd.vp[s][i] = base_mp.vp[s][i]
    end
    ea_fd.patches = base_mp.patches;
    ea_fd.tile_sources = base_mp.tile_sources;
    ea_fd.active_sources = base_mp.active_sources;
    ea_fd
end

