#!/usr/bin/env julia

# This script post-processes the VB results to produce test scores
# and csv files.

using Celeste
using CelesteTypes

import WCSLIB
using DataFrames


const color_names = ["$(band_letters[i])$(band_letters[i+1])" for i in 1:4]


function load_celeste_predictions(model_dir, stamp_id)
    f = open("$model_dir/$(ARGS[1])-$stamp_id.dat")
    mp = deserialize(f)
    close(f)
	mp
end


type DistanceException <: Exception
end


function center_obj(vp::Vector{Vector{Float64}})
	distances = [norm(vs[ids.mu] .- 51/2) for vs in vp]
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
    col_names = ["stamp_id", "pos1", "pos2", "is_star", "flux_r", "flux_r_sd",
            color_col_names, color_sd_col_names,
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


function load_photo_obj!(i::Int64, stamp_id::String, is_s82::Bool, df::DataFrame)
    blob = SDSS.load_stamp_blob(ENV["STAMP"], stamp_id)
    cat_df = is_s82 ?
        SDSS.load_stamp_catalog_df(ENV["STAMP"], "s82-$stamp_id", blob) :
        SDSS.load_stamp_catalog_df(ENV["STAMP"], stamp_id, blob, match_blob=true)
    cat_ce = is_s82 ?
        SDSS.load_stamp_catalog(ENV["STAMP"], "s82-$stamp_id", blob) :
        SDSS.load_stamp_catalog(ENV["STAMP"], stamp_id, blob, match_blob=true)
    ce, ce_df = center_obj(cat_ce, cat_df)

    df[i, :pos1] = ce.pos[1]
    df[i, :pos2] = ce.pos[2]
    df[i, :is_star] = ce.is_star ? 1. : 0.

    fluxes = ce.is_star ? ce.star_fluxes : ce.gal_fluxes
    df[i, :flux_r] = fluxes[3]
    for c in 1:4
        cc = symbol("color_$(color_names[c])")
        cc_sd = symbol("color_$(color_names[c])_sd")
        if fluxes[c] > 0 && fluxes[c + 1] > 0  # leave as NA otherwise
            df[i, cc] = -2.5log10(fluxes[c] / fluxes[c + 1])
        end
    end

    if !is_s82
        df[i, :gal_fracdev] = ce.gal_frac_dev
        df[i, :gal_ab] = ce.gal_ab
        df[i, :gal_angle] = (180/pi)ce.gal_angle
        df[i, :gal_scale] = ce.gal_scale
    else
        # only record the truth, when it's comparable to both sets of predictions
        if ce.is_star < .5
            if !(0.05 < ce.gal_frac_dev < 0.95)
                df[i, :gal_fracdev] = ce.gal_frac_dev
            end

            if !(0.05 < ce.gal_frac_dev < 0.95) || 
                    abs(ce_df[1, :ab_dev] - ce_df[1, :ab_exp]) < 0.1 # proportion
                df[i, :gal_ab] = ce.gal_ab
            end

            if (ce.gal_ab < .6) &&
                (!(0.05 < ce.gal_frac_dev < 0.95) ||
                    abs(ce_df[1, :phi_dev] - ce_df[1, :phi_exp]) < 10)  # degrees
                df[i, :gal_angle] = (180/pi)ce.gal_angle
            end

            if !(0.05 < ce.gal_frac_dev < 0.95) ||
                    abs(ce_df[1, :theta_dev] - ce_df[1, :theta_exp]) < 0.2  # arcsec
                df[i, :gal_scale] = ce.gal_scale
            end
        end
    end
end


function load_synth_obj!(i::Int64, stamp_id::String, df::DataFrame)
    blob = SDSS.load_stamp_blob(ENV["STAMP"], stamp_id)
    f = open(ENV["STAMP"]"/cat-synth-$stamp_id.dat")
    cat_ce = deserialize(f)
    close(f)
    cat_df = SDSS.load_stamp_catalog_df(ENV["STAMP"], "s82-$stamp_id", blob)
    ce, ce_df = center_obj(cat_ce, cat_df)

    df[i, :pos1] = ce.pos[1]
    df[i, :pos2] = ce.pos[2]
    df[i, :is_star] = ce.is_star ? 1. : 0.

    fluxes = ce.is_star ? ce.star_fluxes : ce.gal_fluxes
    df[i, :flux_r] = fluxes[3]
    for c in 1:4
        cc = symbol("color_$(color_names[c])")
        cc_sd = symbol("color_$(color_names[c])_sd")
        if fluxes[c] > 0 && fluxes[c + 1] > 0  # leave as NA otherwise
            df[i, cc] = -2.5log10(fluxes[c] / fluxes[c + 1])
        end
    end

    df[i, :gal_fracdev] = ce.gal_frac_dev
    df[i, :gal_ab] = ce.gal_ab
    df[i, :gal_angle] = (180/pi)ce.gal_angle
    df[i, :gal_scale] = ce.gal_scale
end


function load_celeste_obj!(i::Int64, stamp_id::String, df::DataFrame)
    blob = SDSS.load_stamp_blob(ENV["STAMP"], stamp_id)
    mp = load_celeste_predictions(ENV["MODEL"], stamp_id)
    vs = center_obj(mp.vp)

    df[i, :pos1] = vs[ids.mu][1]
    df[i, :pos2] = vs[ids.mu][2]
    df[i, :is_star] = 1. - vs[ids.chi]

    j = vs[ids.chi] < .5 ? 1 : 2
    df[i, :flux_r] = vs[ids.gamma[j]] * vs[ids.zeta[j]]
    df[i, :flux_r_sd] = sqrt(df[i, :flux_r] * vs[ids.zeta[j]])

    for c in 1:4
        cc = symbol("color_$(color_names[c])")
        cc_sd = symbol("color_$(color_names[c])_sd")
        df[i, cc] = 2.5 * log10(e) * vs[ids.beta[c, j]]
        df[i, cc_sd] = 2.5 * log10(e) * vs[ids.lambda[c, j]]
    end

    df[i, :gal_fracdev] = vs[ids.theta]
    df[i, :gal_ab] = vs[ids.rho]
    df[i, :gal_angle] = (180/pi)vs[ids.phi]
    df[i, :gal_scale] = vs[ids.sigma]
end


function load_df(stamp_ids, per_stamp_callback::Function)
    N = length(stamp_ids)
    df = init_results_df(stamp_ids)

    for i in 1:N
        stamp_id = stamp_ids[i]
        df[i, :stamp_id] = stamp_id
        try
            per_stamp_callback(i, stamp_id, df)
        catch ex
            println("Exception for stamp $stamp_id")
            isa(ex, DistanceException) ? 
                println("No center object in stamp $stamp_id") : throw(ex)
        end
    end

    df
end


function degrees_to_diff(a, b)
    angle_between = abs(a - b) % 180
    min(angle_between, 180 - angle_between)
end


function get_err_df(truth::DataFrame, predicted::DataFrame)
    color_cols = [symbol("color_$cn") for cn in color_names]
    abs_err_cols = [:flux_r, color_cols, :gal_fracdev, :gal_ab, :gal_scale]
    col_symbols = [:stamp_id, :position, :missed_stars, :missed_gals, abs_err_cols, :gal_angle]
            
    col_types = Array(DataType, length(col_symbols))
    fill!(col_types, Float64)
    col_types[1] = String
    col_types[[3,4]] = Bool
    ret = DataFrame(col_types, size(truth, 1))
    names!(ret, col_symbols)
    ret[:stamp_id] = truth[:stamp_id]
    ret

    for n in abs_err_cols
        ret[n] = abs(predicted[n] - truth[n])
    end

    predicted_gal = convert(BitArray, predicted[:is_star] .< .5)
    true_gal = convert(BitArray, truth[:is_star] .< .5)
    ret[:missed_stars] =  predicted_gal & !(true_gal)
    ret[:missed_gals] =  !predicted_gal & true_gal

    ret[:position] = sqrt((truth[:pos1] - predicted[:pos1]).^2 
            + (truth[:pos2] - predicted[:pos2]).^2)

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
    coadd_callback(i, stamp_id, df) = ARGS[1] == "V" ?
        load_photo_obj!(i, stamp_id, true, df) :
        load_synth_obj!(i, stamp_id, df)
    coadd_df = load_df(stamp_ids, coadd_callback)
    primary_callback(i, stamp_id, df) = load_photo_obj!(i, stamp_id, false, df)
    primary_df = load_df(stamp_ids, primary_callback)
    celeste_df = load_df(stamp_ids, load_celeste_obj!)

    primary_err = get_err_df(coadd_df, primary_df)
    celeste_err = get_err_df(coadd_df, celeste_df)

    ttypes = [Symbol, Float64, Float64, Float64, Float64, Int64]
    scores_df = DataFrame(ttypes, length(names(celeste_err)) - 1)
    names!(scores_df, [:field, :primary, :primary_sd, :celeste, :celeste_sd, :N])
    for i in 1:(size(celeste_err, 2) - 1)
        n = names(celeste_err)[i + 1]
        if n == :stamp_id
            continue
        end
        good_row = !isna(primary_err[:, n]) & !isna(celeste_err[:, n])
		if sum(good_row) == 0
			continue
		end
        celeste_mean_err = mean(celeste_err[good_row, n])
        scores_df[i, :field] = n
        scores_df[i, :N] = sum(good_row)
        scores_df[i, :primary] = mean(primary_err[good_row, n])
        scores_df[i, :celeste] = mean(celeste_err[good_row, n])
		if sum(good_row) > 1
			scores_df[i, :primary_sd] = std(primary_err[good_row, n]) / sqrt(scores_df[i, :N])
			scores_df[i, :celeste_sd] = std(celeste_err[good_row, n]) / sqrt(scores_df[i, :N])
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


if length(ARGS) >= 2
    # The input file is the stamp ids you want to process.
    # It looks for models and raw data using the environment variables
    # $STAMP and $MODELS.  Missing stamp ids will not be handled gracefully.
    f = open(ARGS[2])
    stamp_ids = [strip(line) for line in readlines(f)]
    close(f)

    println(df_score(stamp_ids))
end

