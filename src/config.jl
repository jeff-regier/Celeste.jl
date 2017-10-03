# Configuration parameters
struct Config
    # A minimum pixel radius to be included around each source.
    min_radius_pix::Float64

    # number of temperatures (for annealed importance sampling)
    num_ais_temperatures::Int64

    # number of independent AIS samples
    num_ais_samples::Int64
end

Config() = Config(8.0, 50, 10)
Config(min_radius_pix) = Config(min_radius_pix, 50, 10)
