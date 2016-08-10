module ModelInit

export sample_prior, cat_init, peak_init

using Distributions

using ..Util
using ..CelesteTypes


function sample_prior()
    Phi = 0.5

    const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

    Upsilon = Array(Float64, 2)
    Psi = Array(Float64, 2)
    r_file = open("$dat_dir/r_prior.dat")
    ((Upsilon[1], Psi[1]), (Upsilon[2], Psi[2])) = deserialize(r_file)
    close(r_file)

    Xi = Array(Vector{Float64}, 2)
    Omega = Array(Array{Float64, 2}, 2)
    Lambda = Array(Array{Array{Float64, 2}}, 2)
    ck_file = open("$dat_dir/ck_prior.dat")
    ((Xi[1], Omega[1], Lambda[1]), (Xi[2], Omega[2], Lambda[2])) = deserialize(ck_file)
    close(r_file)

    PriorParams(Phi, Upsilon, Psi, Xi, Omega, Lambda)
end


#TODO: use blob (and perhaps priors) to initialize these sensibly
function init_source(init_pos::Vector{Float64})
    ret = Array(Float64, length(all_params))
    ret[ids.chi] = 0.5
    ret[ids.mu[1]] = init_pos[1]
    ret[ids.mu[2]] = init_pos[2]
    ret[ids.gamma] = 1e3
    ret[ids.zeta] = 2e-3
    ret[ids.theta] = 0.5
    ret[ids.rho] = 0.5
    ret[ids.phi] = 0.
    ret[ids.sigma] = 1.
    ret[ids.kappa] = 1. / size(ids.kappa, 1)
    ret[ids.beta] = 0.
    ret[ids.lambda] =  1e-2
    ret
end


function init_source(ce::CatalogEntry)
    ret = init_source(ce.pos)

    ret[ids.gamma[1]] = max(0.0001, ce.star_fluxes[3]) ./ ret[ids.zeta[1]]
    ret[ids.gamma[2]] = max(0.0001, ce.gal_fluxes[3]) ./ ret[ids.zeta[2]]

    get_color(c2, c1) = begin
        c2 > 0 && c1 > 0 ? min(max(log(c2 / c1), -9.), 9.) :
            c2 > 0 && c1 <= 0 ? 3.0 :
                c2 <= 0 && c1 > 0 ? -3.0 : 0.0
    end
    get_colors(raw_fluxes) = begin
        [get_color(raw_fluxes[c+1], raw_fluxes[c]) for c in 1:4]
    end

    ret[ids.beta[:, 1]] = get_colors(ce.star_fluxes)
    ret[ids.beta[:, 2]] = get_colors(ce.gal_fluxes)

    ret[ids.theta] = min(max(ce.gal_frac_dev, 0.01), 0.99)

    ret[ids.rho] = ce.is_star ? .8 : min(max(ce.gal_ab, 0.0001), 0.9999)
    ret[ids.phi] = ce.gal_angle
    ret[ids.sigma] = ce.is_star ? 0.2 : max(ce.gal_scale, 0.2)

    ret
end


function cat_init(cat::Vector{CatalogEntry}; patch_radius::Float64=Inf,
        tile_width::Int64=typemax(Int64))
    vp = [init_source(ce) for ce in cat]
    # TODO: use non-trivial patch radii, based on the catalog
    patches = [SkyPatch(ce.pos, patch_radius) for ce in cat]
    ModelParams(vp, sample_prior(), patches, tile_width)
end


end
