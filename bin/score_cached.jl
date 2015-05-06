#!/usr/bin/env julia

# This script post-processes the VB results to produce test scores
# and csv files.

using Celeste
using CelesteTypes

import WCSLIB
import Base.convert
using DataFrames


const color_names = ["$(band_letters[i])$(band_letters[i+1])" for i in 1:4]


function load_celeste_predictions(model_dir, stamp_id)
    f = open("$model_dir/$(ARGS[2])-$stamp_id.dat")
    mp = deserialize(f)
    close(f)
	mp
end


type DistanceException <: Exception
end


function center_obj(vp::Vector{Vector{Float64}})
	distances = [norm(vs[ids.u] .- 51/2) for vs in vp]
	s = findmin(distances)[2]
    if distances[s] > 2.
        throw(DistanceException())
    end
    vp[s]
end


function center_obj(catalog_ce::Vector{CatalogEntry}, catalog_df::DataFrame)
    @assert length(catalog_ce) == size(catalog_df, 1)
	distances = [norm(ce.pos .- 26.) for ce in catalog_ce]
	idx = findmin(distances)[2]
    catalog_ce[idx], catalog_df[idx,:]
end


function init_results_df(stamp_ids)
    N = length(stamp_ids)
    color_col_names = ["color_$cn" for cn in color_names]
    color_sd_col_names = ["color_$(cn)_sd" for cn in color_names]
    col_names = ["stamp_id", "pos1", "pos2", "is_star", 
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
    col_types[1] = String
    df = DataFrame(col_types, N)
    names!(df, col_symbols)
    df[:stamp_id] = stamp_ids
    df
end


function load_ce!(i::Int64, ce::CatalogEntry, stamp_id::String, df::DataFrame)
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


function load_photo_obj!(i::Int64, stamp_id::String, 
            is_s82::Bool, is_synth::Bool, df::DataFrame)
    blob = SDSS.load_stamp_blob(ENV["STAMP"], stamp_id)
    cat_df = is_s82 ?
        SDSS.load_stamp_catalog_df(ENV["STAMP"], "s82-$stamp_id", blob) :
        SDSS.load_stamp_catalog_df(ENV["STAMP"], stamp_id, blob, match_blob=true)
    cat_ce = is_synth ? begin
            f = open(ENV["STAMP"]"/cat-synth-$stamp_id.dat")
            the_cat_ce = deserialize(f)
            close(f)
            the_cat_ce
        end : is_s82 ?
        SDSS.load_stamp_catalog(ENV["STAMP"], "s82-$stamp_id", blob) :
        SDSS.load_stamp_catalog(ENV["STAMP"], stamp_id, blob, match_blob=true)
    ce, ce_df = center_obj(cat_ce, cat_df)
    load_ce!(i, ce, stamp_id, df)
end


function convert(::Type{CatalogEntry}, vs::Vector{Float64})
    function get_fluxes(i::Int64)
        ret = Array(Float64, 5)
        ret[3] = vs[ids.r1[i]] * vs[ids.r2[i]]
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
        vs[ids.e_scale])
end


function load_celeste_obj!(i::Int64, stamp_id::String, df::DataFrame)
    mp = load_celeste_predictions(ARGS[1], stamp_id)
    vs = center_obj(mp.vp)
    ce = convert(CatalogEntry, vs)
    load_ce!(i, ce, stamp_id, df)

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


function load_df(stamp_ids, per_stamp_callback::Function)
    N = length(stamp_ids)
    df = init_results_df(stamp_ids)

    for i in 1:N
        stamp_id = stamp_ids[i]
        df[i, :stamp_id] = stamp_id
        per_stamp_callback(i, stamp_id, df)
    end

    df
end


function degrees_to_diff(a, b)
    angle_between = abs(a - b) % 180
    min(angle_between, 180 - angle_between)
end


function get_err_df(truth::DataFrame, predicted::DataFrame)
    color_cols = [symbol("color_$cn") for cn in color_names]
    abs_err_cols = [:gal_fracdev, :gal_ab, :gal_scale]
    col_symbols = [:stamp_id, :position, :missed_stars, :missed_gals, 
        :flux_r, color_cols, abs_err_cols, :gal_angle]
            
    col_types = Array(DataType, length(col_symbols))
    fill!(col_types, Float64)
    col_types[1] = String
    col_types[[3,4]] = Bool
    ret = DataFrame(col_types, size(truth, 1))
    names!(ret, col_symbols)
    ret[:stamp_id] = truth[:stamp_id]

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

    ret[:gal_angle] = degrees_to_diff(truth[:gal_angle], predicted[:gal_angle])

    ret
end


function print_latex_table(df)
    for i in 1:size(df, 1)
        is_num_wrong = (df[i, :field] in [:missed_stars, :missed_gals])::Bool
        @printf("%-12s & %.2f (%.2f) & %.2f (%.2f) & %d \\\\\n",
            df[i, :field],
            df[i, :primary] * (is_num_wrong ? df[i, :N] : 1.),
            df[i, :primary_sd],
            df[i, :celeste] * (is_num_wrong ? df[i, :N] : 1.),
            df[i, :celeste_sd],
            df[i, :N])
    end
    println("")
end


function df_score(stamp_ids)
    coadd_callback(i, stamp_id, df) = ARGS[2] == "V" ?
        load_photo_obj!(i, stamp_id, true, false, df) :
        load_photo_obj!(i, stamp_id, true, true, df)
    coadd_df = load_df(stamp_ids, coadd_callback)
    primary_callback(i, stamp_id, df) = load_photo_obj!(i, stamp_id, false, false, df)
    primary_df = load_df(stamp_ids, primary_callback)
    celeste_df = load_df(stamp_ids, load_celeste_obj!)

    primary_err = get_err_df(coadd_df, primary_df)
    celeste_err = get_err_df(coadd_df, celeste_df)

#=
    abs(ce_df[1, :ab_dev] - ce_df[1, :ab_exp]) < 0.1 # proportion
    abs(ce_df[1, :phi_dev] - ce_df[1, :phi_exp]) < 10  # degrees
    abs(ce_df[1, :theta_dev] - ce_df[1, :theta_exp]) < 0.2  # arcsec
=#

    ttypes = [Symbol, Float64, Float64, Float64, Float64, Int64]
    scores_df = DataFrame(ttypes, length(names(celeste_err)) - 1)
    names!(scores_df, [:field, :primary, :celeste, :diff, :diff_sd, :N])
    for i in 1:(size(celeste_err, 2) - 1)
        n = names(celeste_err)[i + 1]
        if n == :stamp_id
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

    if length(ARGS) >= 3 && ARGS[3] == "--csv"
        writetable("coadd.csv", coadd_df)
        writetable("primary.csv", primary_df)
        writetable("celeste.csv", celeste_df)
    end
    if length(ARGS) >= 3 && ARGS[3] == "--latex"
		print_latex_table(scores_df)
	end
    scores_df
end


function filter_filenames(rx, filenames)
    s_filenames = filter((fn)->ismatch(rx, fn), filenames)
    String[match(rx, fn).captures[1] for fn in s_filenames]
end


if length(ARGS) >= 2
    filenames = readdir(ARGS[1])
    s_ids = filter_filenames(r"S-(.*)\.dat", filenames)
    v_ids = filter_filenames(r"V-(.*)\.dat", filenames)
    stamp_ids = intersect(s_ids, v_ids)

    println(df_score(stamp_ids))
end

