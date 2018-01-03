using Celeste
using Celeste.SDSSIO

@testset "main" begin

@testset "read_config" begin
    config, datasets = Celeste.read_config("configs/plain.yml")
    @test datasets["sdss"] == SDSSDataSet("data")

    config, datasets = Celeste.read_config("configs/notplain.yml")
    @test datasets["sdss"] == SDSSDataSet("data";
                                          dirlayout = :sdss,
                                          iostrategy = :masterrpc,
                                          compressed = true,
                                          slurp = true)
end


@testset "executable" begin
    # ensure data exists locally
    SampleData.fetch_sdss_data(3900, 6, 269, "all")

    Celeste.main(["sdss", "164.39", "164.41", "39.11", "39.13",
                  "--config", "configs/plain.yml"])
end

end
