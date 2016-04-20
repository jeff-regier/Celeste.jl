import Base.convert

import ModelInit.ModelParams

function convert(FDType::Type{ForwardDiff.GradientNumber},
                 mp::ModelParams{Float64})
    x = mp.vp[1]
    P = length(x)
    FDType = ForwardDiff.GradientNumber{length(mp.vp[1]), Float64}

    fd_x = [ ForwardDiff.GradientNumber(x[i], zeros(Float64, P)...) for i=1:P ]
    convert(FDType, x[1])

    vp_fd = convert(Array{Array{FDType, 1}, 1}, mp.vp[1])
    mp_fd = ModelParams(vp_fd, mp.pp)
end

function convert(FDType::Type{ForwardDiff.HessianNumber},
                 mp::ModelParams{Float64})
    x = mp.vp[1]
    P = length(x)
    FDType = ForwardDiff.HessianNumber{length(mp.vp[1]), Float64}

    fd_x = [ ForwardDiff.HessianNumber(x[i], zeros(Float64, P)...) for i=1:P ]
    convert(FDType, x[1])

    vp_fd = convert(Array{Array{FDType, 1}, 1}, mp.vp[1])
    mp_fd = ModelParams(vp_fd, mp.pp)
end


# TODO: Maybe write it as a convert()?
function forward_diff_model_params{T <: Number}(
            FDType::Type{T},
            base_mp::ModelParams{Float64})
    S = length(base_mp.vp)
    P = length(base_mp.vp[1])
    mp_fd = ModelParams{FDType}([ zeros(FDType, P) for s=1:S ], base_mp.pp);
    # Set the values (but not gradient numbers) for parameters other
    # than the galaxy parameters.
    for s=1:base_mp.S, i=1:length(ids)
        mp_fd.vp[s][i] = base_mp.vp[s][i]
    end
    mp_fd.patches = base_mp.patches;
    mp_fd.tile_sources = base_mp.tile_sources;
    mp_fd.active_sources = base_mp.active_sources;
    mp_fd.objids = base_mp.objids;
    mp_fd
end

