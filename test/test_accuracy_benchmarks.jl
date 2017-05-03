using Base.Test

using DataFrames
import FITSIO

import Celeste: AccuracyBenchmark
import Celeste: DeterministicVI
import Celeste: Model

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
    @test isna(AccuracyBenchmark.color_from_fluxes(15.0, 0.0))
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

@testset "variational params -> data frame -> catalog entry conversion" begin
    variational_params = DeterministicVI.generic_init_source([1.0, 2.0])
    variational_params[Model.ids.e_axis] = 0.5
    variational_params[Model.ids.e_scale] = 10.0
    variational_params[Model.ids.e_angle] = -pi / 4
    variational_params[Model.ids.r1[2]] = log(20.0)
    variational_params[Model.ids.a[1]] = 0.01
    variational_params[Model.ids.a[2]] = 0.99

    data = AccuracyBenchmark.variational_parameters_to_data_frame_row("12345", variational_params)
    # we'll just check a few particularly troublesome parameters :)
    @test isapprox(data[1, :half_light_radius_px], 10 * sqrt(0.5))
    @test isapprox(data[1, :angle_deg], 135.0)
    @test isapprox(data[1, :reference_band_flux_nmgy], 20.0)

    data[1, :is_star] = false
    catalog_entry = AccuracyBenchmark.make_catalog_entry(first(eachrow(data)))
    @test isapprox(catalog_entry.gal_scale, 10.0)
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
                is_saturated=false,
                half_light_radius_px=10.0,
                de_vaucouleurs_mixture_weight=0.99,
                minor_major_axis_ratio=0.8,
            ),
            DataFrame(
                is_saturated=false,
                minor_major_axis_ratio=0.5,
                angle_deg=10.0,
                declination_deg=0.0,
            )
        )
    end

    truth, error = make_data()

    function check_row(column_name=:minor_major_axis_ratio)
        AccuracyBenchmark.is_good_row(first(eachrow(truth)), first(eachrow(error)), column_name)
    end

    @test check_row()

    truth, error = make_data()
    error[1, :minor_major_axis_ratio] = NaN
    @test !check_row()

    truth, error = make_data()
    error[1, :minor_major_axis_ratio] = NA
    @test !check_row()

    truth, error = make_data()
    truth[1, :half_light_radius_px] = NA
    @test check_row()

    truth, error = make_data()
    truth[1, :half_light_radius_px] = 25.0
    @test !check_row()

    truth, error = make_data()
    truth[1, :is_saturated] = true
    @test !check_row()

    truth, error = make_data()
    error[1, :is_saturated] = true
    @test !check_row()

    truth, error = make_data()
    truth[1, :de_vaucouleurs_mixture_weight] = 0.5
    @test !check_row()
    @test check_row(:declination_deg)

    truth, error = make_data()
    @test !check_row(:angle_deg)
    truth[1, :minor_major_axis_ratio] = 0.2
    @test check_row(:angle_deg)
end

@testset "sky distance" begin
    @test isapprox(AccuracyBenchmark.sky_distance_px(0.0, 0.0, 0.02, 0.01), 203.27, atol=0.01)
    @test isapprox(AccuracyBenchmark.sky_distance_px(0.0, 70.0, 0.02, 70.01), 110.14, atol=0.01)

    ras = [0.04, 0.02, 0.06]
    decs = [0.04, 0.01, 0.06]
    matching_indices = AccuracyBenchmark.match_position(ras, decs, 0.0, 0.0, 210)
    @test matching_indices == [2]
    matching_indices = AccuracyBenchmark.match_position(ras[[1,3]], decs[[1,3]], 0.0, 0.0, 150)
    @test isempty(matching_indices)
end
