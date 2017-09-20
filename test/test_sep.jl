import Celeste: SEP
using Base.Test
using FITSIO

@testset "sep" begin
    
    # use test image in SEP source directory
    sep_testdata_dir = joinpath(dirname(@__FILE__), "..", "deps", "src",
                                "sep-1.0.2", "data")

    data = read(FITS(joinpath(sep_testdata_dir, "image.fits"))[1])
    back_sextractor = read(FITS(joinpath(sep_testdata_dir, "back.fits"))[1])
    rms_sextractor = read(FITS(joinpath(sep_testdata_dir, "rms.fits"))[1])

    # test background
    bkg = SEP.Background(data)
    @test collect(bkg) ≈ back_sextractor

    for T in (Float32, Float64)
        A = zeros(T, size(data))
        back = collect(T, bkg)
        @test back ≈ back_sextractor
    end

    # test that broadcast subtraction methods work and return same result
    A = zeros(Float32, size(data))
    A .-= bkg
    B = zeros(Float32, size(data)) .- bkg
    C = zeros(Float32, size(data)) - bkg
    @test A == B == C

    # test source extraction
    data .-= bkg
    catalog = SEP.extract(data, 3.0; noise=SEP.rms(bkg))
    @test length(catalog.x) == 39
end
