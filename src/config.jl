# Configuration parameters
struct Config
    # A minimum pixel radius to be included around each source.
    min_radius_pix::Float64

    # number of temperatures (for annealed importance sampling)
    num_ais_temperatures::Int

    # number of independent AIS samples
    num_ais_samples::Int

    # Number of iterations in joint VI. One iteration optimizes a full
    # pass over target sources.
    num_joint_vi_iters::Int
end

function Config(; min_radius_pix=8.0,
                num_ais_temperatures=50,
                num_ais_samples=10,
                num_joint_vi_iters=3)
    Config(min_radius_pix,
           num_ais_temperatures,
           num_ais_samples,
           num_joint_vi_iters)
end
