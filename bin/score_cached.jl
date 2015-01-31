#!/usr/bin/env julia

using Celeste
using CelesteTypes

import WCSLIB
using DataFrames


const color_names = ["$(band_letters[i])$(band_letters[i+1])" for i in 1:4]


function load_celeste_predictions(stamp_id)
    f = open(ENV["STAMP"]"/V-$stamp_id.dat")
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



function init_results_df(N::Int64)
    color_col_names = ["color_$cn" for cn in color_names]
    color_sd_col_names = ["color_$(cn)_sd" for cn in color_names]
    col_names = ["stamp_id", "ra", "dec", "is_star", "flux_r", "flux_r_sd",
            color_col_names, color_sd_col_names,
            "gal_fracdev", "gal_ab", "gal_angle", "gal_scale"]
    col_symbols = Symbol[symbol(cn) for cn in col_names]
    col_types = Array(DataType, length(col_names))
    fill!(col_types, Float64)
    col_types[1] = String
    df = DataFrame(col_types, N)
    names!(df, col_symbols)
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

    df[i, :ra] = ce_df[1, :ra]
    df[i, :dec] = ce_df[1, :dec]
    df[i, :is_star] = ce_df[1, :is_star] ? 1. : 0.

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

    if !(0.05 < ce.gal_frac_dev < 0.95) || 
            abs(ce_df[1, :ab_dev] - ce_df[1, :ab_exp]) < 0.1 # proportion
        df[i, :gal_ab] = ce.gal_ab
    end

    if (ce.gal_ab < .6) &&
        (!(0.05 < ce.gal_frac_dev < 0.95) ||
            abs(ce_df[1, :phi_dev] - ce_df[1, :phi_exp]) < 10)  # degrees
        df[i, :gal_angle] = ce.gal_angle
    end

    if !(0.05 < ce.gal_frac_dev < 0.95) ||
            abs(ce_df[1, :theta_dev] - ce_df[1, :theta_exp]) < 0.2  # arcsec
        df[i, :gal_scale] = ce.gal_scale
    end
end


function load_celeste_obj!(i::Int64, stamp_id::String, df::DataFrame)
    blob = SDSS.load_stamp_blob(ENV["STAMP"], stamp_id)
    mp = load_celeste_predictions(stamp_id)
    vs = center_obj(mp.vp)

    ra_dec = WCSLIB.wcsp2s(blob[3].wcs, vs[ids.mu]'')

    df[i, :ra] = ra_dec[1]
    df[i, :dec] = ra_dec[2]
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

    df[i, :gal_fracdev] = vs[ids.chi]
    df[i, :gal_ab] = vs[ids.rho]

    my_angle = pi/2 - vs[ids.phi]
    my_angle -= floor(my_angle / pi) * pi
    my_angle *= 180 / pi
    df[i, :gal_angle] = my_angle

    df[i, :gal_scale] = vs[ids.sigma] * 0.396
end


function load_df(stamp_ids, per_stamp_callback::Function)
    N = length(stamp_ids)
    df = init_results_df(N)

    for i in 1:N
        stamp_id = stamp_ids[i]
        df[i, :stamp_id] = stamp_id
        try
            per_stamp_callback(i, stamp_id, df)
        catch ex
            isa(ex, DistanceException) ? 
                println("No center object in stamp $stamp_id") : throw(ex)
        end
    end

    df
end


function print_comparison(quantity_name, true_val, base_val, my_val)
    println("\n$quantity_name:")
    println("truth: $true_val")
    println("Photo: $base_val")
    println("Celeste: $my_val")
end


function load_predictions(stamp_id)
    blob = SDSS.load_stamp_blob(ENV["STAMP"], stamp_id)
    true_cat_df = SDSS.load_stamp_catalog_df(ENV["STAMP"], "s82-$stamp_id", blob)
    true_cat = SDSS.load_stamp_catalog(ENV["STAMP"], "s82-$stamp_id", blob)
    baseline_cat_df = SDSS.load_stamp_catalog_df(ENV["STAMP"], stamp_id, blob,
         match_blob=true)
    baseline_cat = SDSS.load_stamp_catalog(ENV["STAMP"], stamp_id, blob,
         match_blob=true)
    mp = load_celeste_predictions(stamp_id)

    true_ce, true_row = center_obj(true_cat, true_cat_df)
    base_ce, base_row = center_obj(baseline_cat, baseline_cat_df)
	vs = center_obj(mp.vp)

    true_ce, true_row, base_ce, base_row, vs
end



function report_on_stamp(stamp_id)
    true_ce, true_row, base_ce, base_row, vs = load_predictions(stamp_id)

    println("================= STAMP $stamp_id ====================")

    print_comparison("position (pixel coordinates)", 
        true_ce.pos, round(base_ce.pos, 3), round(vs[ids.mu], 3))

    print_comparison("celestial body type",
        true_ce.is_star ? "star" : "galaxy", 
        base_ce.is_star ? "star" : "galaxy",
        vs[ids.chi] < .5 ? 
            "star ($(100 - 100vs[ids.chi])% certain)" :
            "galaxy ($(100vs[ids.chi])% certain)")

    if true_ce.is_star
        E_r = vs[ids.gamma[1]] * vs[ids.zeta[1]]
        sd_r = sqrt(E_r * vs[ids.zeta[1]])
        print_comparison("brightness (r-band flux)",
            round(true_ce.star_fluxes[3], 3),
            round(base_ce.star_fluxes[3], 3),
            @sprintf("%.3f (%.3f)", E_r, sd_r))

        for c in 1:4
            true_color = true_ce.star_fluxes[c + 1] <= 0 || true_ce.star_fluxes[c] <= 0. ?
                "NA" : round(log(true_ce.star_fluxes[c + 1] / true_ce.star_fluxes[c]), 3)
            base_color = base_ce.star_fluxes[c + 1] <= 0 || base_ce.star_fluxes[c] <= 0 ?
                "NA" : round(log(base_ce.star_fluxes[c + 1] / base_ce.star_fluxes[c]), 3)
            print_comparison("color $(color_names[c])",
                true_color,
                base_color,
                @sprintf("%.3f (%.3f)", vs[ids.beta[c, 1]], sqrt(vs[ids.lambda[c, 1]])))
        end
    else
        E_r = vs[ids.gamma[2]] * vs[ids.zeta[2]]
        sd_r = sqrt(E_r * vs[ids.zeta[2]])
        print_comparison("flux (r-band)",
            round(true_ce.gal_fluxes[3], 3),
            round(base_ce.gal_fluxes[3], 3),
            @sprintf("%.3f (%.3f)", E_r, sd_r))

        for c in 1:4
            true_color = true_ce.gal_fluxes[c + 1] <= 0 || true_ce.gal_fluxes[c] <= 0. ?
                "NA" : round(log(true_ce.gal_fluxes[c + 1] / true_ce.gal_fluxes[c]), 3)
            base_color = base_ce.gal_fluxes[c + 1] <= 0 || base_ce.gal_fluxes[c] <= 0 ?
                "NA" : round(log(base_ce.gal_fluxes[c + 1] / base_ce.gal_fluxes[c]), 3)
            print_comparison("color $(color_names[c])",
                true_color,
                base_color,
                @sprintf("%.3f (%.3f)", vs[ids.beta[c, 2]], sqrt(vs[ids.lambda[c, 2]])))
        end

        print_comparison("proportion De Vaucouleurs",
            round(true_ce.gal_frac_dev, 3), 
            round(base_ce.gal_frac_dev, 3), 
            round(vs[ids.theta], 3))

        print_comparison("minor/major axis ratio",
            @sprintf("%.3f (dev), %.3f (exp)",
                true_row[1, :ab_dev], true_row[1, :ab_exp]),
            @sprintf("%.3f (dev), %.3f (exp)", 
                base_row[1, :ab_dev], base_row[1, :ab_exp]),
            @sprintf("%.3f (both)", vs[ids.rho]))

        my_angle = pi/2 - vs[ids.phi]
        my_angle -= floor(my_angle / pi) * pi
        my_angle *= 180 / pi
        ab_warning = vs[ids.rho] > .9 ? "   [NOTE: predicted galaxy is nearly isotropic]" : ""
        print_comparison("angle (degrees)",
            @sprintf("%.3f (dev), %.3f (exp)",
                true_row[1, :phi_dev] - floor(true_row[1, :phi_dev] / 180) * 180,
                true_row[1, :phi_exp] - floor(true_row[1, :phi_exp] / 180) * 180),
            @sprintf("%.3f (dev), %.3f (exp)",
                base_row[1, :phi_dev] - floor(base_row[1, :phi_dev] / 180) * 180,
                base_row[1, :phi_exp] - floor(base_row[1, :phi_exp] / 180) * 180),
            string(round(my_angle, 3), " (both)", ab_warning))

        my_er = vs[ids.sigma] * 0.396
        print_comparison("effective radius (arcseconds)",
            @sprintf("%.3f (dev), %.3f (exp)",
                true_row[1, :theta_dev], true_row[1, :theta_exp]),
            @sprintf("%.3f (dev), %.3f (exp)",
                base_row[1, :theta_dev], base_row[1, :theta_exp]),
            @sprintf("%.3f (both)", my_er))
    end

    println("\n")
end


function write_csvs(stamp_ids)
    coadd_callback(i, stamp_id, df) = load_photo_obj!(i, stamp_id, true, df)
    coadd_df = load_df(stamp_ids, coadd_callback)

    primary_callback(i, stamp_id, df) = load_photo_obj!(i, stamp_id, false, df)
    primary_df = load_df(stamp_ids, primary_callback)

    celeste_df = load_df(stamp_ids, load_celeste_obj!)

    writetable("coadd.csv", coadd_df)
    writetable("primary.csv", primary_df)
    writetable("celeste.csv", celeste_df)
end

function degrees_to_diff(a, b)
    angle_between = abs(a - b) % 180
    min(angle_between, 180 - angle_between)
end


function score_stamps(stamp_ids)
    N = length(stamp_ids)

    pos_err = zeros(2, N)
    obj_type_err = falses(2, N)

    flux_r_err = zeros(2, N)

    num_na = zeros(4)
    color_err = zeros(2, N, 4)

    num_fracdev = 0
    gal_frac_dev_err = zeros(2, N)

    num_ab = 0
    gal_ab_err = zeros(2, N)

    num_angle = 0
    gal_angle_err = zeros(2, N)

    num_scale = 0
    gal_scale_err = zeros(2, N)

    N2 = 0
    function process_one_stamp(i)
        stamp_id = stamp_ids[i]
        true_ce, true_row, base_ce, base_row, vs = load_predictions(stamp_id)

        # doesn't count if an exception happens above
        N2 += 1

        pos_err[1, i] = norm(true_ce.pos - base_ce.pos)
        pos_err[2, i] = norm(true_ce.pos - vs[ids.mu])

        obj_type_err[1, i] = true_ce.is_star != base_ce.is_star
        obj_type_err[2, i] = true_ce.is_star != (vs[ids.chi] < .5)

        true_fluxes = true_ce.is_star ? true_ce.star_fluxes : true_ce.gal_fluxes
        base_fluxes = base_ce.is_star ? base_ce.star_fluxes : base_ce.gal_fluxes
        flux_r_err[1, i] = abs(base_fluxes[3] - true_fluxes[3])
        j = vs[ids.chi] < .5 ? 1 : 2
        celeste_r_flux = vs[ids.gamma[j]] * vs[ids.zeta[j]]
        flux_r_err[2, i] = abs(celeste_r_flux - true_fluxes[3])

        for c in 1:4
            if true_fluxes[c + 1] <= 0 || true_fluxes[c] <= 0. ||
                base_fluxes[c + 1] <= 0 || base_fluxes[c] <= 0.
                println("NA for color $c, stamp_id: $stamp_id")
                num_na[c] += 1
                color_err[1, i, c] = color_err[2, i, c] = 0
            else
                true_color = log(true_fluxes[c + 1] ./ true_fluxes[c])
                base_color = log(base_fluxes[c + 1] ./ base_fluxes[c])
                color_err[1, i, c] = abs(true_color - base_color)
                color_err[2, i, c] = abs(true_color - vs[ids.beta[c, j]])
            end
        end

        if !true_ce.is_star
            if true_ce.gal_frac_dev > .95 || true_ce.gal_frac_dev < 0.05
                num_fracdev += 1
                gal_frac_dev_err[1, i] = abs(true_ce.gal_frac_dev - base_ce.gal_frac_dev)
                gal_frac_dev_err[2, i] = abs(true_ce.gal_frac_dev - vs[ids.theta])
            end

            if true_ce.gal_frac_dev > .95 || true_ce.gal_frac_dev < 0.05 ||
                   abs(true_row[1, :ab_dev] - true_row[1, :ab_exp]) < .2
                num_ab += 1
                gal_ab_err[1, i] = abs(true_ce.gal_ab - base_ce.gal_ab)
                gal_ab_err[2, i] = abs(true_ce.gal_ab - vs[ids.rho])
            end

            if (true_ce.gal_frac_dev > .95 || true_ce.gal_frac_dev < 0.05 ||
               abs(true_row[1, :phi_dev] - true_row[1, :phi_exp]) < 10) &&
                    true_ce.gal_ab < .6 && base_ce.gal_ab < .6 &&
                    vs[ids.rho] < .6
                num_angle += 1
                true_deg = (180/pi)true_ce.gal_angle
                base_deg = (180/pi)base_ce.gal_angle
                celeste_deg = (180/pi)vs[ids.phi]
                gal_angle_err[1, i] = degrees_to_diff(true_deg, base_deg)
                gal_angle_err[2, i] = degrees_to_diff(true_deg, celeste_deg)
            end

            if true_ce.gal_frac_dev > .95 || true_ce.gal_frac_dev < 0.05 ||
                   abs(true_row[1, :theta_dev] - true_row[1, :theta_exp]) < .2
                num_scale += 1
                gal_scale_err[1, i] = abs(true_ce.gal_scale - base_ce.gal_scale) * .396
                gal_scale_err[2, i] = abs(true_ce.gal_scale - vs[ids.sigma]) * .396
            end
        end
    end


    for i in 1:N
        try
            process_one_stamp(i)
        catch ex
            if isa(ex, DistanceException)
                println("No center object in stamp $i")
            else
                throw(ex)
            end
        end
    end

    println("N:$N N2:$N2  num_fracdev:$num_fracdev  num_ab:$num_ab",
        "num_angle:$num_angle  num_scale:$num_scale")

    println("pos err: ", sum(pos_err, 2)[:] ./ N2)
    println("obj type err: ", sum(obj_type_err, 2)[:])
    println("flux r err: ", sum(flux_r_err, 2)[:] / N2)

    for c in 1:4
        println("color $(color_names[c]) err: ", 
            sum(color_err[:, :, c], 2)[:] / (N2 - num_na[c]))
    end

    println("frac_dev err: ", sum(gal_frac_dev_err, 2)[:] / num_fracdev)
    println("ab err: ", sum(gal_ab_err, 2)[:] / num_ab)
    println("angle err: ", sum(gal_angle_err, 2)[:] / num_angle)
    println("scale err: ", sum(gal_scale_err, 2)[:] / num_scale)
end


f = open(ARGS[2])
stamp_ids = [strip(line) for line in readlines(f)]
close(f)

if ARGS[1] == "--report"
    for stamp_id in stamp_ids
        try
            report_on_stamp(stamp_id)
        catch ex
            if isa(ex, DistanceException)
                println("No center object in $stamp_id")
            else
                throw(ex)
            end
        end
    end
elseif ARGS[1] == "--score"
    score_stamps(stamp_ids)
elseif ARGS[1] == "--csv"
    write_csvs(stamp_ids)
end

