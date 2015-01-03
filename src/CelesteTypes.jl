# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module CelesteTypes

export CatalogEntry, CatalogStar, CatalogGalaxy

export Image, Blob, PriorParams
export PsfComponent, GalaxyComponent, GalaxyPrototype
export galaxy_prototypes

export SourceParam, SourceParams, VariationalParams
export ParamStruct, ModelParams, convert, zero_source_param
export SourceMatrix, zero_source_matrix
export zero_all_param, const_all_param, AllParam
export values, deriv, SensitiveParam, zero_all_matrix
export AllParamMatrix, ParamIndex, get_dim
export clear_param!, accum_all_param!
export full_index, getindex, Patch

import Base.convert
import Base.show

import FITSIO
import Distributions
import WCSLIB



abstract CatalogEntry

type CatalogStar <: CatalogEntry
    mu::Vector{Float64}
    gamma::Vector{Float64}
end

type CatalogGalaxy <: CatalogEntry
    mu::Vector{Float64}
    zeta::Vector{Float64}
    theta::Float64
    Xi::Vector{Float64}
end


############################################


immutable GalaxyComponent
	alphaTilde::Float64
	sigmaTilde::Float64
end


typealias GalaxyPrototype Vector{GalaxyComponent}

function get_galaxy_prototypes()
	exp_amp = [
		2.34853813e-03, 3.07995260e-02, 2.23364214e-01,
		1.17949102e+00, 4.33873750e+00, 5.99820770e+00]
	exp_amp /= sum(exp_amp)
	exp_var = [
		1.20078965e-03, 8.84526493e-03, 3.91463084e-02,
		1.39976817e-01, 4.60962500e-01, 1.50159566e+00]
	exp_prototype = [GalaxyComponent(exp_amp[j], exp_var[j]) for j in 1:6]

	dev_amp = [
		4.26347652e-02, 2.40127183e-01, 6.85907632e-01, 1.51937350e+00,
		2.83627243e+00, 4.46467501e+00, 5.72440830e+00, 5.60989349e+00]
	dev_amp /= sum(dev_amp)
	dev_var = [
		2.23759216e-04, 1.00220099e-03, 4.18731126e-03, 1.69432589e-02,
		6.84850479e-02, 2.87207080e-01, 1.33320254e+00, 8.40215071e+00]
	dev_prototype = [GalaxyComponent(dev_amp[j], dev_var[j]) for j in 1:8]
	(exp_prototype, dev_prototype)
end

const galaxy_prototypes = get_galaxy_prototypes()


immutable PsfComponent
	alphaBar::Float64  # TODO: use underscore
	xiBar::Vector{Float64}
	SigmaBar::Matrix{Float64}

	SigmaBarInv::Matrix{Float64}
	SigmaBarLd::Float64

	PsfComponent(alphaBar::Float64, xiBar::Vector{Float64}, SigmaBar::Matrix{Float64}) = begin
		new(alphaBar, xiBar, SigmaBar, SigmaBar^-1, logdet(SigmaBar))
	end
end

immutable Patch
	center::Vector{Float64}
	radius::Float64
end


type Image
	H::Int64
	W::Int64
	pixels::Matrix{Float64}
	b::Int64
	wcs::WCSLIB.wcsprm
	epsilon::Float64
	psf::Vector{PsfComponent}
end


typealias Blob Vector{Image}


immutable PriorParams
	rho::Float64	
	Delta::Matrix{Float64}
	Theta::Vector{Float64}
	Lambda::Matrix{Float64}
end

#########################################################

immutable ParamStruct{T}  # use macros instead?
	chi::T
	mu::(T,T)
	gamma::(T,T,T,T,T)
	tau::T
	zeta::(T,T,T,T,T)
	theta::T
	Xi::(T,T,T)
end

function getindex(ps::ParamStruct, i::Int64)
	field_lengths = (1, 2, 5, 1, 5, 1, 3)
	if 0 < i <= 1 ps.chi
    elseif i <= 3 ps.mu[i - 1]
    elseif i <= 8 ps.gamma[i - 3]
    elseif i <= 9 ps.tau
    elseif i <= 14 ps.zeta[i - 9]
	elseif i <= 15 ps.theta
    elseif i <= 18 ps.Xi[i - 15]
	end
end 

function convert{T}(::Type{Vector{T}}, ps::ParamStruct{T})
	T[ps[p] for p in 1:18]
end

function convert{T}(::Type{ParamStruct{T}}, x::Vector{T})
	@assert(length(x) == 18)
	ParamStruct{T}(x[1], (x[2], x[3]), 
		(x[4], x[5], x[6], x[7], x[8]), x[9],
		(x[10], x[11], x[12], x[13], x[14]), x[15],
		(x[16], x[17], x[18]))
end

const param_names = names(ParamStruct)
const param_ids_struct = ParamStruct{Int64}(1, (2, 3), (4,5,6,7,8), 9, (10,11,12,13,14), 15, (16,17,18))

typealias ParamIndex Vector{Int64}

const full_index = [1:18]

function get_dim(index::ParamIndex)
	length(index)
end


#########################################################

type SensitiveParam{S, T}
    v::S
    d::T
	index::ParamIndex
end

#########################################################

typealias SourceParam SensitiveParam{Float64, Vector{Float64}}

#########################################################

typealias SourceParams ParamStruct{Float64}

function show(vs::SourceParams)
    string("chi: $(vs.chi)\n",
        "mu: $(vs.mu)\n",
        "gamma: $(vs.gamma)\n",
        "tau: $(vs.tau)\n",
        "zeta: $(vs.zeta)\n",
        "theta: $(vs.theta)\n",
        "Xi: $(vs.Xi)\n")
end

function zero_source_param(index::ParamIndex)
	SourceParam(0., zeros(get_dim(index)), index)
end

function clear_param!(sp::SensitiveParam)
	sp.v = 0.
	fill!(sp.d, 0.)
end

#########################################################

type ModelParams
	vp::Vector{ParamStruct{Float64}}
	pp::PriorParams
	patches::Vector{Patch}
	S::Int64

	ModelParams(vp, pp, patches) = begin
		@assert length(vp) == length(patches)
		new(vp, pp, patches, length(vp))
	end
end

#########################################################

typealias AllParam SensitiveParam{Float64, Matrix{Float64}}

function const_all_param(S::Int64, c::Float64, index::ParamIndex)
	AllParam(c, zeros(get_dim(index), S), index)
end

function zero_all_param(S::Int64, index::ParamIndex)
	const_all_param(S, 0., index)
end

function accum_all_param!(src::AllParam, accum::AllParam)
	 # add source index to vary # sources
	@assert(size(accum.d, 2) == size(src.d, 2))
	@assert(size(accum.d, 1) == 18)
	accum.v += src.v
	for s in 1:size(src.d, 2)
		for src_p in 1:size(src.d, 1)
			global_p = src.index[src_p]
			accum.d[global_p, s] += src.d[src_p, s]
		end
	end	
end


#########################################################

typealias SourceMatrix SensitiveParam{Matrix{Float64}, Array{Float64, 3}}

function convert(::Type{SourceMatrix}, mat::Matrix{SourceParam})
    SourceMatrix(values(mat), deriv(mat))
end

function convert(::Type{Matrix{SourceParam}}, sm::SourceMatrix)
    ret = Array(SourceParam, size(sm.v))
    for w in 1:size(sm.v)[2], h in 1:size(sm.v)[1]
        ret[h, w] = SourceParam(sm.v[h, w], sm.d[h, w, :][:]) #invert it if slow
    end
    ret
end

function zero_source_matrix(H::Int64, W::Int64, index::ParamIndex)
	D = get_dim(index)
    SourceMatrix(zeros(H, W), zeros(D, H, W), index)
end

########################################################

typealias AllParamMatrix SensitiveParam{Array{Float64, 2}, Array{Float64, 4}}

function zero_all_matrix(H::Int64, W::Int64, S::Int64, index::ParamIndex)
	D = get_dim(index)
	AllParamMatrix(zeros(H, W), zeros(D, H, W, S), index)
end

############# accessors ###################

function values{T <: SensitiveParam}(arr::Array{T})
	map((x)->x.v, arr)
end

function values(arr::Array{Float64})
	arr
end

function values(x::SensitiveParam)
	x.v
end

function values(x::Float64)
	x
end

function values(arr::Array{Float64})
	arr
end

function deriv{T <: SensitiveParam}(arr::Array{T})
	map((x)->x.d, arr)
end


end

