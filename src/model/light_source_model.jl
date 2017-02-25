import JLD


# The number of components in the color prior.
const D = 8

# The number of types of celestial objects (here, stars and galaxies).
const Ia = 2


type CatalogEntry
    pos::Vector{Float64}
    is_star::Bool
    star_fluxes::Vector{Float64}
    gal_fluxes::Vector{Float64}
    gal_frac_dev::Float64
    gal_ab::Float64
    gal_angle::Float64
    gal_scale::Float64
    objid::String
    thing_id::Int
end


"""
Parameters of a single normal component of a galaxy.

Attributes:
  etaBar: The weight of the galaxy component
  nuBar: The scale of the galaxy component
"""
immutable GalaxyComponent
    etaBar::Float64
    nuBar::Float64
end


@compat const GalaxyPrototype = Vector{GalaxyComponent}


"""
Pre-defined shapes for galaxies.

Returns:
  dev_prototype: An array of GalaxyComponent for de Vaucouleurs galaxy types
  exp_prototype: An array of GalaxyComponent for exponenttial galaxy types
"""
function get_galaxy_prototypes()
    dev_amp = [
        4.26347652e-2, 2.40127183e-1, 6.85907632e-1, 1.51937350,
        2.83627243, 4.46467501, 5.72440830, 5.60989349]
    dev_amp /= sum(dev_amp)
    dev_var = [
        2.23759216e-4, 1.00220099e-3, 4.18731126e-3, 1.69432589e-2,
        6.84850479e-2, 2.87207080e-1, 1.33320254, 8.40215071]

    exp_amp = [
        2.34853813e-3, 3.07995260e-2, 2.23364214e-1,
        1.17949102, 4.33873750, 5.99820770]
    exp_amp /= sum(exp_amp)
    exp_var = [
        1.20078965e-3, 8.84526493e-3, 3.91463084e-2,
        1.39976817e-1, 4.60962500e-1, 1.50159566]

    # Adjustments to the effective radius hard-coded above.
    # (The effective radius is the distance from the center containing half
    # the light.)
    effective_radii = [1.078031, 0.928896]
    dev_var /= effective_radii[1]^2
    exp_var /= effective_radii[2]^2

    exp_prototype = [GalaxyComponent(exp_amp[j], exp_var[j]) for j in 1:6]
    dev_prototype = [GalaxyComponent(dev_amp[j], dev_var[j]) for j in 1:8]
    (dev_prototype, exp_prototype)
end


const galaxy_prototypes = get_galaxy_prototypes()


immutable PriorParams
    a::Vector{Float64}  # formerly Phi
    r_μ::Vector{Float64}
    r_σ²::Vector{Float64}
    k::Matrix{Float64}  # formerly Xi
    c_mean::Array{Float64, 3} # formerly Omega
    c_cov::Array{Float64, 4} # formerly Lambda
    e_scale_μ::Float64
    e_scale_σ²::Float64
end


function load_prior()
    # set a = [.99, .01] if stars are underrepresented
    # due to the greater flexibility of the galaxy model
    #a = [0.28, 0.72]
    a = [0.099, 0.001]
    k = Matrix{Float64}(D, Ia)
    c_mean = Array{Float64}(B - 1, D, Ia)
    c_cov = Array{Float64}(B - 1, B - 1, D, Ia)

    prior_params = [JLD.load(joinpath(cfgdir, "star_prior.jld")),
                    JLD.load(joinpath(cfgdir, "gal_prior.jld"))]
    for i in 1:2
        k[:, i] = prior_params[i]["c_weights"]
        c_mean[:, :, i] = prior_params[i]["c_means"]
        c_cov[:, :, :, i] = prior_params[i]["c_covs"]
    end

    # log normal parameters for the r-band brightness prior.
    # these were fit by maximum likelihood to the output of primary
    # on one field.
    r_μ = Float64[1.5035546, 1.07431]
    r_σ² = Float64[1.9039063^2, 1.1177502^2]

    # If I remove this next statement, compile time for
    # benchmark_infer.jl jumps from 13 seconds to 300 seconds!
    # Really. It's crazy!
    log(42.)

    # Compile time is still over 20x as long if `log(42.)`
    # is replaced with `exp(2.)`. Insane! Julia 0.5.0 mac os x.
    # Update: happens on Linux with 0.5.0 binaries too.
    # exp(2.)

    # log normal prior parameters (location, scale) on galaxy scale.
    # determined by fitting a univariate log normal to primary's
    # output the region of stripe 82 we use for validation
    e_scale_μ = 0.5015693
    e_scale_σ² = 0.8590007^2

    PriorParams(a, r_μ, r_σ², k, c_mean, c_cov, e_scale_μ, e_scale_σ²)
end


const prior = load_prior()
