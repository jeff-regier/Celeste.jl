using Celeste

const read_with_strategy = function(strategy)
    rcf = Celeste.RunCamcolField(4263, 5, 119)
    Celeste.SDSSIO.load_field_images(strategy, rcf)
    Celeste.SDSSIO.load_field_catalog(strategy, rcf)
end

function test_config(config)
    strategy = Celeste.read_settings_file(joinpath("ioconfigs", config))
    read_with_strategy(strategy)
end

@testset "io" begin
    for config in ["plain.yml", "slurp.yml"]
        test_config(config)
    end

    rm("data/dr12", force = true, recursive=true)
    cd("data") do
        # Create the original SDSS directory layout for testing purposes
        mkpath("dr12/boss/photo/redux/301/4263/objcs/5")
        mkpath("dr12/boss/photoObj/frames/301/4263/5")
        mkpath("dr12/boss/photoObj/301/4263/5")
        for band in ['z','g','r','i','u']
            cp("4263/5/119/frame-$band-004263-5-0119.fits", "dr12/boss/photoObj/frames/301/4263/5/frame-$band-004263-5-0119.fits")
            cp("4263/5/119/fpM-004263-$(band)5-0119.fit", "dr12/boss/photo/redux/301/4263/objcs/5/fpM-004263-$(band)5-0119.fit")
        end
        cp("4263/5/119/psField-004263-5-0119.fit", "dr12/boss/photo/redux/301/4263/objcs/5/psField-004263-5-0119.fit")
        cp("4263/5/119/photoObj-004263-5-0119.fits", "dr12/boss/photoObj/301/4263/5/photoObj-004263-5-0119.fits")
        cp("4263/5/photoField-004263-5.fits", "dr12/boss/photoObj/301/4263/photoField-004263-5.fits")
    end

    test_config("sdsslayout.yml")

    rm("data/dr12", recursive=true)

    worker = addprocs(1)[]
    fetch(@spawnat worker Base.require(:Celeste))

    # Test fetching over the network The basedir is relative, if the
    # worker doesn't try to fetch via the network after this, it'll
    # fail
    fetch(@spawnat worker cd("/tmp"))
    remotecall_fetch(read_with_strategy, worker, Celeste.SDSSIO.MasterRPCStrategy(Celeste.read_settings_file(joinpath("ioconfigs", "plain.yml"))))
    rmprocs([worker])
end
