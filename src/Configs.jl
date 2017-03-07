module Configs

type Config
    # A minimum pixel radius to be included around each source.
    min_radius_pix::Float64

    function Config()
        config = new()
        config.min_radius_pix = 8.0
        config
    end
end

end
