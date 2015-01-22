# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module CelesteTypes

export CatalogEntry, CatalogStar, CatalogGalaxy

export Image, Blob, SkyPatch, ImageTile, PsfComponent
export GalaxyComponent, GalaxyPrototype, galaxy_prototypes

export ModelParams, PriorParams, VariationalParams

export SensitiveFloat
export zero_sensitive_float, const_sensitive_param, clear!, accum!

export ParamIndex, ids, all_params, star_pos_params, galaxy_pos_params, D

import FITSIO
import Distributions
import WCSLIB


abstract CatalogEntry

type CatalogStar <: CatalogEntry
    pos::Vector{Float64}
    fluxes::Vector{Float64}
end

type CatalogGalaxy <: CatalogEntry
    pos::Vector{Float64}
    fluxes::Vector{Float64}
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

	PsfComponent(alphaBar::Float64, xiBar::Vector{Float64}, 
			SigmaBar::Matrix{Float64}) = begin
		new(alphaBar, xiBar, SigmaBar, SigmaBar^-1, logdet(SigmaBar))
	end
end

type Image
	H::Int64
	W::Int64
	pixels::Matrix{Float64}
	b::Int64
	wcs::WCSLIB.wcsprm
	epsilon::Float64
	iota::Float64
	psf::Vector{PsfComponent}
end

immutable ImageTile
	hh::Int64 # tile coordinates---not pixel or sky coordinates
	ww::Int64
	img::Image
end

typealias Blob Vector{Image}

immutable SkyPatch #pixel coordinates for now, soon wcs
	center::Vector{Float64}
	radius::Float64
end


#########################################################

immutable PriorParams
	Delta::Float64	
	Upsilon::Vector{Float64}
	Phi::Vector{Float64}
    Psi::Vector{Vector{Float64}}
    Omega::Vector{Array{Float64, 2}}
    Lambda::Vector{Array{Array{Float64, 2}}}
end

# TODO: use a matrix here, in conjunction with ArrayViews.jl (?)
typealias VariationalParams Vector{Vector{Float64}}

type ModelParams
	vp::VariationalParams
	pp::PriorParams
	patches::Vector{SkyPatch}
	tile_width::Int64
	S::Int64

	ModelParams(vp, pp, patches, tile_width) = begin
		@assert length(vp) == length(patches)
		new(vp, pp, patches, tile_width, length(vp))
	end
end

#########################################################

immutable ParamIndex
	chi::Int64
	mu::Vector{Int64}
	gamma::Vector{Int64}
	zeta::Vector{Int64}
	theta::Int64
	Xi::Vector{Int64}
	kappa::Array{Int64, 2}
	beta::Array{Int64, 2}
	lambda::Array{Int64, 2}
end

const D = 16

function get_param_ids()
	I = 2
	B = 5

	kappa_end = 11 + I * D
	beta_end = kappa_end + I * (B - 1)
	lambda_end = beta_end + I * (B - 1)

	kappa_ids = reshape([12 : kappa_end], D, I)
	beta_ids = reshape([kappa_end + 1 : beta_end], B - 1, I)
	lambda_ids = reshape([beta_end + 1 : lambda_end], B - 1, I)

	ParamIndex(1, [2, 3], [4, 5], [6, 7], 8, [9, 10, 11], 
			kappa_ids, beta_ids, lambda_ids)
end

const ids = get_param_ids()

const all_params = [1:ids.lambda[end]]
const star_pos_params = ids.mu
const galaxy_pos_params = [ids.mu, ids.theta, ids.Xi]

#########################################################

type SensitiveFloat
    v::Float64
    d::Matrix{Float64} # local_P x local_S
	source_index::Vector{Int64}
	param_index::Vector{Int64}
end

#########################################################

function zero_sensitive_float(s_index::Vector{Int64}, p_index::Vector{Int64})
	d = zeros(length(p_index), length(s_index))
	SensitiveFloat(0., d, s_index, p_index)
end

function clear!(sp::SensitiveFloat)
	sp.v = 0.
	fill!(sp.d, 0.)
end

function accum!(src::SensitiveFloat, accum::SensitiveFloat)
	accum.v += src.v
	for child_s in 1:size(src.d, 2)
		parent_s = src.source_index[child_s]
		#parent_s and parent_p aren't necessarily global indexes
		for child_p in 1:size(src.d, 1)
			parent_p = src.index[child_p]
			accum.d[global_p, tile_s] += src.d[child_p, child_s]
		end
	end	
end

#########################################################

end

