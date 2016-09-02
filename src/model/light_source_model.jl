
# The number of components in the color prior.
const D = 2

# The number of types of celestial objects (here, stars and galaxies).
const Ia = 2


type CatalogEntry
    pos::Vec{2, Float64}
    is_star::Bool
    star_fluxes::Vec{5, Float64}
    gal_fluxes::Vec{5, Float64}
    gal_frac_dev::Float64
    gal_ab::Float64
    gal_angle::Float64
    gal_scale::Float64
    objid::Vec{30, UInt8}
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


typealias GalaxyPrototype Vector{GalaxyComponent}


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
    r_mean::Vector{Float64}
    r_var::Vector{Float64}
    k::Matrix{Float64}  # formerly Xi
    c_mean::Array{Float64, 3} # formerly Omega
    c_cov::Array{Float64, 4} # formerly Lambda
end


function load_prior()
    # set a = [.99, .01] if stars are underrepresented
    # due to the greater flexibility of the galaxy model
    #a = [0.28, 0.72]
    a = [0.99, 0.01]
    r_mean = Array(Float64, Ia)
    r_var = Array(Float64, Ia)
    k = Array(Float64, D, Ia)
    c_mean = Array(Float64, B - 1, D, Ia)
    c_cov = Array(Float64, B - 1, B - 1, D, Ia)

    v05 = VERSION >= v"0.5.0-dev" ? "-v05" : ""
    stars_file = open(joinpath(cfgdir, "stars$D$v05.dat"))
    r_fit1, k[:, 1], c_mean[:,:,1], c_cov[:,:,:,1] = deserialize(stars_file)
    close(stars_file)

    gals_file = open(joinpath(cfgdir, "gals$D$v05.dat"))
    r_fit2, k[:, 2], c_mean[:,:,2], c_cov[:,:,:,2] = deserialize(gals_file)
    close(gals_file)

    # These "magic numbers" have been in use for a while.
    # They were initially gamma parameters, and now they are log normal
    # parameters.  TODO: Get rid of these and use an empirical prior.
    # r = [0.47 1.28; 1/0.012 1/0.11] # These were gamma (shape, scale)

    mean_brightness = [0.47 / 0.012, 1.28 / 0.11 ]
    var_brightness = [0.47 / (0.012 ^ 2), 1.28 / (0.11 ^ 2) ]

    # The prior contains parameters of a lognormal distribution with
    # the desired means.
    r_var = log(var_brightness ./ (mean_brightness .^ 2) + 1)
    r_mean = log(mean_brightness) - 0.5 * r_var
    PriorParams(a, r_mean, r_var, k, c_mean, c_cov)
end


prior = load_prior()


"""
Return a default-initialized VariationalParams object.
"""
function init_source(init_pos::Vector{Float64})
    ret = Array(Float64, length(CanonicalParams))
    ret[ids.a[2, 1]] = 0.5
    ret[ids.a[1, 1]] = 1.0 - ret[ids.a[2, 1]]
    ret[ids.u[1]] = init_pos[1]
    ret[ids.u[2]] = init_pos[2]
    ret[ids.r1] = log(2.0)
    ret[ids.r2] = 1e-3
    ret[ids.e_dev] = 0.5
    ret[ids.e_axis] = 0.5
    ret[ids.e_angle] = 0.
    ret[ids.e_scale] = 1.
    ret[ids.k] = 1. / size(ids.k, 1)
    ret[ids.c1] = 0.
    ret[ids.c2] =  1e-2
    ret
end


"""
Return a VariationalParams object initialized form a catalog entry.
"""
function init_source(ce::CatalogEntry)
    # TODO: sync this up with the transform bounds
    ret = init_source(ce.pos)

    ret[ids.a[1, 1]] = ce.is_star ? 0.8: 0.2
    ret[ids.a[2, 1]] = ce.is_star ? 0.2: 0.8

    ret[ids.r1[1]] = log(max(0.1, ce.star_fluxes[3]))
    ret[ids.r1[2]] = log(max(0.1, ce.gal_fluxes[3]))

    function get_color(c2, c1)
        c2 > 0 && c1 > 0 ? min(max(log(c2 / c1), -9.), 9.) :
            c2 > 0 && c1 <= 0 ? 3.0 :
                c2 <= 0 && c1 > 0 ? -3.0 : 0.0
    end

    function get_colors(raw_fluxes)
        [get_color(raw_fluxes[c+1], raw_fluxes[c]) for c in 1:4]
    end

    ret[ids.c1[:, 1]] = get_colors(ce.star_fluxes)
    ret[ids.c1[:, 2]] = get_colors(ce.gal_fluxes)

    ret[ids.e_dev] = min(max(ce.gal_frac_dev, 0.015), 0.985)

    ret[ids.e_axis] = ce.is_star ? .8 : min(max(ce.gal_ab, 0.015), 0.985)
    ret[ids.e_angle] = ce.gal_angle
    ret[ids.e_scale] = ce.is_star ? 0.2 : max(ce.gal_scale, 0.2)

    ret
end
