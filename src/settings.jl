using YAML
using Celeste.SDSSIO

# For now this is only used for IO settings, but other settings could use this too
function read_settings_file(file)
    data = open(YAML.load, file)
    ios = data["io"]
    dataset = get(ios, "dataset", "sdss")
    if dataset != "sdss"
        error("Only SDSS is supported at the moment")
    end
    stagedir = get(ios, "basedir", get(ENV, "CELESTE_STAGE_DIR", ""))
    isempty(stagedir) && error("No data directory set")
    strategy = get(ios, "strategy", "fits")
    if strategy == "fits"
        dirlayout = Symbol(get(ios, "dirlayout", "celeste"))
        slurp = get(ios, "slurp", false)
        compressed = get(ios, "compressed", false)
        return Celeste.SDSSIO.PlainFITSStrategy(stagedir, dirlayout, compressed, stagedir, slurp)
    elseif strategy == "bigfiles"
        error("TODO")
    end
end
