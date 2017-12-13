using Base.Test

using DataFrames
import FITSIO

import Celeste: AccuracyBenchmark
import Celeste: DeterministicVI
import Celeste: Model

@testset "accuracy benchmarks" begin

@testset "whole accuracy benchmark pipeline runs" begin
    images = SampleData.get_sdss_images(4263, 5, 119)

    rcf = RunCamcolField(4263, 5, 119)
    primary_df = AccuracyBenchmark.load_primary(rcf, SampleData.DATADIR)
    output_path = joinpath(SampleData.DATADIR, "test_primary.csv")
    new_csv = AccuracyBenchmark.write_catalog(output_path, primary_df)
    primary_df2 = AccuracyBenchmark.read_catalog(new_csv)
    catalog_entries = AccuracyBenchmark.make_initialization_catalog(primary_df2, true)
    # entry 8 is a star near [0.513037, 0.535631],
    # see http://legacysurvey.org/viewer/jpeg-cutout/?ra=0.5130&dec=0.5358&zoom=16&layer=sdss2
    target_sources = [8]

    patches = Model.get_sky_patches(images, catalog_entries)
    neighbor_map = Dict(i=>Model.find_neighbors(patches, i)
                        for i in target_sources)
    results = ParallelRun.one_node_single_infer(catalog_entries,
                                                patches,
                                                target_sources,
                                                neighbor_map, images,
                                                config=Celeste.Config())
    results_df = AccuracyBenchmark.celeste_to_df(results)

    coadd_path = joinpath(SampleData.DATADIR, "coadd_for_4263_5_119.fit")
    coadd_df = AccuracyBenchmark.load_coadd_catalog(coadd_path)
    coadd_path2 = joinpath(SampleData.DATADIR, "test_coadd.csv")
    coadd_csv = AccuracyBenchmark.write_catalog(coadd_path2, coadd_df)
    coadd_df2 = AccuracyBenchmark.read_catalog(coadd_csv)

    scores = AccuracyBenchmark.score_predictions(coadd_df2, [results_df])

    uncertainty_df = AccuracyBenchmark.get_uncertainty_df(coadd_df2, results_df)
    uq_scores = AccuracyBenchmark.score_uncertainty(uncertainty_df)
end

@testset "flux <-> mags conversion" begin
    # reference values based on `nmgy2lups()` from
    # https://github.com/esheldon/sdsspy/blob/683d6e0f16a998240a129942f80ad3ce6e7d5dfe/sdsspy/util.py
    @test isapprox(AccuracyBenchmark.flux_to_mag(15.0, 1), 19.559677, atol=1e-5)
    @test isapprox(AccuracyBenchmark.flux_to_mag(15.0, 3),19.559702, atol=1e-5)
    @test isapprox(AccuracyBenchmark.mag_to_flux(19.559677, 1), 15.0, atol=1e-5)
    @test isapprox(AccuracyBenchmark.mag_to_flux(19.559702, 3), 15.0, atol=1e-5)
end

@testset "color calculations" begin
    @test isapprox(AccuracyBenchmark.color_from_fluxes(15.0, 20.0), log(20 / 15))
    @test ismissing(AccuracyBenchmark.color_from_fluxes(15.0, 0.0))
    fluxes = AccuracyBenchmark.fluxes_from_colors(10.0, [-1.0, 0.0, 1.0, 2.0])
    @test isapprox(fluxes[1], exp(1.0) * 10.0)
    @test isapprox(fluxes[2], 10.0)
    @test isapprox(fluxes[3], 10.0)
    @test isapprox(fluxes[4], exp(1.0) * 10.0)
    @test isapprox(fluxes[5], exp(3.0) * 10.0)
end

@testset "angle canonicalization" begin
    @test AccuracyBenchmark.canonical_angle(95.0) == 95.0
    @test AccuracyBenchmark.canonical_angle(195.0) == 15.0
    @test AccuracyBenchmark.canonical_angle(-20.0) == 160.0

    @test isapprox(AccuracyBenchmark.degrees_to_diff(20., -30.), 50.)
    @test isapprox(AccuracyBenchmark.degrees_to_diff(-10., 190.), 20.)
end

@testset "variational params -> data frame -> catalog entry" begin
    variational_params = DeterministicVI.generic_init_source([1.0, 2.0])
    variational_params[Model.ids.gal_axis_ratio] = 0.5
    variational_params[Model.ids.gal_radius_px] = 10.0
    variational_params[Model.ids.gal_angle] = -pi / 4
    variational_params[Model.ids.flux_loc[2]] = log(20.0)
    variational_params[Model.ids.is_star[1]] = 0.01
    variational_params[Model.ids.is_star[2]] = 0.99

    data = AccuracyBenchmark.variational_parameters_to_data_frame_row(
        variational_params)
    # we'll just check a few particularly troublesome parameters :)
    @test isapprox(data[1, :gal_radius_px], 10 * sqrt(0.5))
    @test isapprox(data[1, :gal_angle_deg], 135.0)
    @test isapprox(data[1, :flux_r_nmgy], 20.0)

    data[1, :is_star] = false
    catalog_entry = AccuracyBenchmark.make_catalog_entry(first(eachrow(data)))
    @test isapprox(catalog_entry.gal_radius_px, 10.0)
    @test isapprox(catalog_entry.gal_angle, 3 * pi / 4)
    @test isapprox(catalog_entry.gal_fluxes[3], 20.0)
end

@testset "PSF serialize/deserialize" begin
    psf = PsfComponent[
        PsfComponent(0.3, StaticArrays.@SVector([1., 2.]), StaticArrays.@SMatrix([3. 1.; 1. 4.])),
        PsfComponent(0.7, StaticArrays.@SVector([-2., -3.]), StaticArrays.@SMatrix([2. 0.; 0. 1.]))
    ]
    header = FITSIO.FITSHeader(String[], [], String[])

    AccuracyBenchmark.serialize_psf_to_header(psf, header)
    new_psf = AccuracyBenchmark.make_psf_from_header(header)

    @test new_psf == psf
end

const FITS_HEADER_STRING = "SIMPLE  =                    T /                                                BITPIX  =                  -32 / 32 bit floating point                          NAXIS   =                    2                                                  NAXIS1  =                 2048                                                  NAXIS2  =                 1489                                                  EXTEND  =                    T /Extensions may be present                       BZERO   =              0.00000 /Set by MRD_SCALE                                BSCALE  =              1.00000 /Set by MRD_SCALE                                TAI     =        4537928038.58 / 1st row - Number of seconds since Nov 17 1858  RA      =            359.76830 / 1st row - Right ascension of telescope boresighDEC     =            0.000000  / 1st row - Declination of telescope boresight (dSPA     =              90.000  / 1st row - Camera col position angle wrt north (IPA     =             102.229  / 1st row - Instrument rotator position angle (deIPARATE =              0.0000  / 1st row - Instrument rotator angular velocity (AZ      =            14.483496 / 1st row - Azimuth  (encoder) of tele (0=N?) (deALT     =            56.247810 / 1st row - Altitude (encoder) of tele        (deFOCUS   =           -436.80000 / 1st row - Focus piston (microns?)              DATE-OBS= '2002-09-05'         / 1st row - TAI date                             TAIHMS  = '07:33:58.57'        / 1st row - TAI time (HH:MM:SS.SS) (TAI-UT = apprCOMMENT  TAI,RA,DEC,SPA,IPA,IPARATE,AZ,ALT,FOCUS at reading of col 0, row 0     ORIGIN  = 'SDSS    '                                                            "

@testset "parse FITS header" begin
    header = AccuracyBenchmark.parse_fits_header_from_string(FITS_HEADER_STRING)
    @test header["SIMPLE"] == true
    @test header["NAXIS1"] == 2048
    @test isapprox(header["RA"], 359.7683)
    @test strip(header["ORIGIN"]) == "SDSS"
end

@testset "filtering rows for scoring errors" begin
    function make_data()
        (
            DataFrame(
                gal_radius_px=Union{Float64, Missing}[10.0],
                gal_frac_dev=Union{Float64, Missing}[0.99],
                gal_axis_ratio=Union{Float64, Missing}[0.8],
            ),
            DataFrame(
                gal_axis_ratio=Union{Float64, Missing}[0.5],
                gal_angle_deg=Union{Float64, Missing}[10.0],
                dec=Union{Float64, Missing}[0.0],
            )
        )
    end

    truth, error = make_data()

    function check_row(column_name=:gal_axis_ratio)
        AccuracyBenchmark.is_good_row(first(eachrow(truth)), first(eachrow(error)), column_name)
    end

    @test check_row()

    truth, error = make_data()
    error[1, :gal_axis_ratio] = NaN
    @test !check_row()

    truth, error = make_data()
    error[1, :gal_axis_ratio] = missing
    @test !check_row()

    truth, error = make_data()
    truth[1, :gal_radius_px] = missing
    @test check_row()

    truth, error = make_data()
    truth[1, :gal_radius_px] = 25.0
    @test !check_row()

    truth, error = make_data()
    truth[1, :gal_frac_dev] = 0.5
    @test !check_row()
    @test check_row(:dec)

    truth, error = make_data()
    @test !check_row(:gal_angle_deg)
    truth[1, :gal_axis_ratio] = 0.2
    @test check_row(:gal_angle_deg)
end


@testset "match catalogs" begin
    ra = [0.0, 1.0, 2.0, 3.0]
    dec = [50.0, 51.0, 52.0, 53.0]
    truth = DataFrame(Any[ra, dec], [:ra, :dec])

    off = 0.2 / 3600.0
    ra1 = [0.0 + off, 1.0 - off, 5.0, 3.0 + off, 4.0]
    dec1 = [50.0 - off, 51.0 + off, 60.0, 53.0 - off, 50.0]
    pred1 = DataFrame(Any[ra1, dec1], [:ra, :dec])

    ra2 = [3.0 - off, 0.0 - off, 7.0, 5.0, 4.0]
    dec2 = [53.0 + off, 50.0 + off, 51.0, 60.0, 50.0]
    pred2 = DataFrame(Any[ra2, dec2], [:ra, :dec])

    truth_matched, predictions_matched =
        AccuracyBenchmark.match_catalogs(truth, [pred1, pred2])

    @test nrow(truth_matched) == 2
    @test length(predictions_matched) == 2
    for pred in predictions_matched
        @test nrow(pred) == 2
    end
    @test truth_matched[:ra] == [0.0, 3.0]
    @test truth_matched[:dec] == [50.0, 53.0]
    @test predictions_matched[1][:ra] == [0.0 + off, 3.0 + off]
    @test predictions_matched[1][:dec] == [50.0 - off, 53.0 - off]
    @test predictions_matched[2][:ra] == [0.0 - off, 3.0 - off]
    @test predictions_matched[2][:dec] == [50.0 + off, 53.0 + off]
end

end
