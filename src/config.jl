# Configuration parameters
struct Config
    # A minimum pixel radius to be included around each source.
    min_radius_pix::Float64
end

Config() = Config(8.0)
