#!/usr/bin/env julia

# This script post-processes the inference results to produce test scores

using Celeste
using CelesteTypes

import Base.convert
using DataFrames

import SloanDigitalSkySurvey: SDSS

const color_names = ["$(band_letters[i])$(band_letters[i+1])" for i in 1:4]


type DistanceException <: Exception
end


"""
Initialize and return a data frame that stores the results for one
method (i.e., Celeste, Primary, or Coadd) for a collection of objects,
in standardized units.
Let's call the type of the returned value a "results data frame".
"""
function init_results_df(object_ids)
    N = length(object_ids)
    color_col_names = ["color_$cn" for cn in color_names]
    color_sd_col_names = ["color_$(cn)_sd" for cn in color_names]
    col_names = ["objid", "pos1", "pos2", "is_star",
            "star_flux_r", "star_flux_r_sd",
            "gal_flux_r", "gal_flux_r_sd",
            ["star_$c" for c in color_col_names],
            ["star_$c" for c in color_sd_col_names],
            ["gal_$c" for c in color_col_names],
            ["gal_$c" for c in color_sd_col_names],
            "gal_fracdev", "gal_ab", "gal_angle", "gal_scale"]
    col_symbols = Symbol[symbol(cn) for cn in col_names]
    col_types = Array(DataType, length(col_names))
    fill!(col_types, Float64)
    col_types[1] = ASCIIString
    df = DataFrame(col_types, N)
    names!(df, col_symbols)
    df[:objid] = object_ids
    df
end


"""
This function loads one catalog entry into row of i of df, a results data
frame.
ce = Catalog Entry, a row of an astronomical catalog
"""
function load_ce!(i::Int64, ce::CatalogEntry, df::DataFrame)
    df[i, :pos1] = ce.pos[1]
    df[i, :pos2] = ce.pos[2]
    df[i, :is_star] = ce.is_star ? 1. : 0.

    for j in 1:2
        s_type = ["star", "gal"][j]
        fluxes = j == 1 ? ce.star_fluxes : ce.gal_fluxes
        df[i, symbol("$(s_type)_flux_r")] = fluxes[3]
        for c in 1:4
            cc = symbol("$(s_type)_color_$(color_names[c])")
            cc_sd = symbol("$(s_type)_color_$(color_names[c])_sd")
            if fluxes[c] > 0 && fluxes[c + 1] > 0  # leave as NA otherwise
                df[i, cc] = -2.5log10(fluxes[c] / fluxes[c + 1])
            end
        end
    end

    df[i, :gal_fracdev] = ce.gal_frac_dev
    df[i, :gal_ab] = ce.gal_ab
    df[i, :gal_angle] = (180/pi)ce.gal_angle
    df[i, :gal_scale] = ce.gal_scale
end


"""
This function converts the parameters from Celeste for one light source
to a CatalogEntry. (which can be passed to load_ce!)
It only needs to be called by load_celeste_obj!
"""
function convert(::Type{CatalogEntry}, vs::Vector{Float64}; objid="converted")
    function get_fluxes(i::Int64)
        ret = Array(Float64, 5)
        ret[3] = exp(vs[ids.r1[i]] + 0.5 * vs[ids.r2[i]])
        ret[4] = ret[3] * exp(vs[ids.c1[3, i]])
        ret[5] = ret[4] * exp(vs[ids.c1[4, i]])
        ret[2] = ret[3] / exp(vs[ids.c1[2, i]])
        ret[1] = ret[2] / exp(vs[ids.c1[1, i]])
        ret
    end

    CatalogEntry(
        vs[ids.u],
        vs[ids.a[1]] > 0.5,
        get_fluxes(1),
        get_fluxes(2),
        vs[ids.e_dev],
        vs[ids.e_axis],
        vs[ids.e_angle],
        vs[ids.e_scale],
        objid)
end


"""
This function is a wrapper around `load_ce!`, for use when loading
the Celeste catalog (rather than Primary or Coadd, which can call load_ce!
directly).

i is the row number in df.
In typical use:
    df = celeste_df
    vs = mp.vp[s] and s is an integer identifying a particular source.
"""
function load_celeste_obj!(i::Int64, vs::Vector{Float64}, df::DataFrame)
    ce = convert(CatalogEntry, vs)
    load_ce!(i, ce, df)

    df[i, :is_star] = vs[ids.a[1]]

    for j in 1:2
        s_type = ["star", "gal"][j]
        df[i, symbol("$(s_type)_flux_r_sd")] =
            sqrt(df[i, symbol("$(s_type)_flux_r")]) * vs[ids.r2[j]]
        for c in 1:4
            cc_sd = symbol("$(s_type)_color_$(color_names[c])_sd")
            df[i, cc_sd] = 2.5 * log10(e) * vs[ids.c2[c, j]]
        end
    end
end


"""
Given two results data frame, one containing ground truth (i.e Coadd)
and one containing predictions (i.e., either Primary of Celeste),
compute an a data frame containing each prediction's error.
(It's not an average of the errors, it's each error.)
Let's call the return type of this function an "error data frame".
"""
function get_err_df(truth::DataFrame, predicted::DataFrame)
    color_cols = [symbol("color_$cn") for cn in color_names]
    abs_err_cols = [:gal_fracdev, :gal_ab, :gal_scale]
    col_symbols = [:objid, :position, :missed_stars, :missed_gals,
        :flux_r, color_cols, abs_err_cols, :gal_angle]

    col_types = Array(DataType, length(col_symbols))
    fill!(col_types, Float64)
    col_types[1] = ASCIIString
    col_types[[3,4]] = Bool
    ret = DataFrame(col_types, size(truth, 1))
    names!(ret, col_symbols)
    ret[:objid] = truth[:objid]

    predicted_gal = convert(BitArray, predicted[:is_star] .< .5)
    true_gal = convert(BitArray, truth[:is_star] .< .5)
    ret[:missed_stars] =  predicted_gal & !(true_gal)
    ret[:missed_gals] =  !predicted_gal & true_gal

    ret[:position] = sqrt((truth[:pos1] - predicted[:pos1]).^2
            + (truth[:pos2] - predicted[:pos2]).^2)

    ret[true_gal, :flux_r] =
        abs(truth[true_gal, :gal_flux_r] - predicted[true_gal, :gal_flux_r])
    ret[!true_gal, :flux_r] =
        abs(truth[!true_gal, :star_flux_r] - predicted[!true_gal, :star_flux_r])

    for cn in color_names
        ret[true_gal, symbol("color_$cn")] =
            abs(truth[true_gal, symbol("gal_color_$cn")] -
                predicted[true_gal, symbol("gal_color_$cn")])
        ret[!true_gal, symbol("color_$cn")] =
            abs(truth[!true_gal, symbol("star_color_$cn")] -
                predicted[!true_gal, symbol("star_color_$cn")])
    end

    for n in abs_err_cols
        ret[n] = abs(predicted[n] - truth[n])
    end

    function degrees_to_diff(a, b)
        angle_between = abs(a - b) % 180
        min(angle_between, 180 - angle_between)
    end

    ret[:gal_angle] = degrees_to_diff(truth[:gal_angle], predicted[:gal_angle])

    ret
end


"""
For the particular field, this method
evaluates cached results for both Celeste and Primary,
with ground truth from Coadd.
The returned value is a data frame.
"""
function score_field(run_num, camcol_num, field_num)
    #TODO: load the object ids for this field, the object ids from
    #Primary (not Coadd) that is.

    coadd_df = init_results_df(object_ids)
    primary_df = init_results_df(object_ids)
    celeste_df = init_results_df(object_ids)

    # TODO: populate coadd_df and primary_df through calls to `load_ce!`
    # TODO: populate celeste_df through calls to 'load_celeste_obj!`

    primary_err = get_err_df(coadd_df, primary_df)
    celeste_err = get_err_df(coadd_df, celeste_df)

    ttypes = [Symbol, Float64, Float64, Float64, Float64, Int64]
    scores_df = DataFrame(ttypes, length(names(celeste_err)) - 1)
    names!(scores_df, [:field, :primary, :celeste, :diff, :diff_sd, :N])
    for i in 1:(size(celeste_err, 2) - 1)
        n = names(celeste_err)[i + 1]
        if n == :objid
            continue
        end
        good_row = !isna(primary_err[:, n]) & !isna(celeste_err[:, n])
        if string(n)[1:5] == "star_"
            good_row &= (coadd_df[:is_star] .> 0.5)
        elseif string(n)[1:4] == "gal_"
            good_row &= (coadd_df[:is_star] .< 0.5)
            if n in [:gal_ab, :gal_scale, :gal_angle, :gal_fracdev]
                good_row &= !(0.05 .< coadd_df[:gal_fracdev] .< 0.95)
            end
            if in == :gal_angle
                good_row &= coadd_df[:gal_ab] .< .6
            end
        end

        if sum(good_row) == 0
            continue
        end
        celeste_mean_err = mean(celeste_err[good_row, n])
        scores_df[i, :field] = n
        scores_df[i, :N] = sum(good_row)
        scores_df[i, :primary] = mean(primary_err[good_row, n])
        scores_df[i, :celeste] = mean(celeste_err[good_row, n])
        if sum(good_row) > 1
            scores_df[i, :diff] = scores_df[i, :primary] - scores_df[i, :celeste]
            scores_df[i, :diff_sd] =
                std(Float64[abs(x) for x in primary_err[good_row, n] - celeste_err[good_row, n]]) / sqrt(sum(good_row))
        end
    end

    scores_df
end


# This script scores predictions by for a particular field
# by calling the script's top-level function, "score_field".
if length(ARGS) == 3
    run_num, camcol_num, field_num = ARGS
    println( score_field(run_num, camcol_num, field_num) )
end

