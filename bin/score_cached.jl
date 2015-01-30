#!/usr/bin/env julia

using Celeste
using CelesteTypes

import WCSLIB
using DataFrames


function load_celeste_predictions(stamp_id)
    f = open(ENV["STAMP"]"/V-$stamp_id.dat")
    mp = deserialize(f)
    close(f)
	mp
end


function center_obj(vp::Vector{Vector{Float64}})
	distances = [norm(vs[ids.mu] .- 51/2) for vs in vp]
	s = findmin(distances)[2]
    @assert(distances[s] < 2.)
    vp[s]
end


function center_obj(catalog::DataFrame)
	distances = [norm([catalog[i, :ra], catalog[i, :dec]] .- 26.) 
        for i in 1:size(catalog,1)]
	idx = findmin(distances)[2]
    catalog[idx, :]
end


function center_obj(catalog::Vector{CatalogEntry})
	distances = [norm(ce.pos .- 26.) for ce in catalog]
	idx = findmin(distances)[2]
    catalog[idx]
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

    true_ce = center_obj(true_cat)
    true_row = center_obj(true_cat_df)
    base_ce = center_obj(baseline_cat)
    base_row = center_obj(baseline_cat_df)
	vs = center_obj(mp.vp)

    true_ce, true_row, base_ce, base_row, vs
end


const color_names = ["$(band_letters[i+1])-$(band_letters[i])" for i in 1:4]


function report_on_stamp(stamp_id)
    println("================= STAMP $stamp_id ====================")

    true_ce, true_row, base_ce, base_row, vs = load_predictions(stamp_id)

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


function score_stamps(stamp_ids)
    N = length(stamp_ids)

    pos_err = Array(Float64, 2, N)
    obj_type_err = Array(Bool, 2, N)

    flux_r_err = Array(Float64, 2, N)
    color_err = Array(Float64, 2, N, 4)
    gal_frac_dev_err = Array(Float64, 2, N)
    gal_ab_err = Array(Float64, 2, N)
    gal_angel_err = Array(Float64, 2, N)
    gal_er_err = Array(Float64, 2, N)
    num_na = zeros(4)

    for i in 1:N
        stamp_id = stamp_ids[i]
        true_ce, true_row, base_ce, base_row, vs = load_predictions(stamp_id)

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

        if true_ce.is_star
        else  # galaxy
        end
    end

    println("pos err: ", mean(pos_err, 2)[:])
    println("obj type err: ", sum(obj_type_err, 2)[:])
    println("flux r err: ", mean(flux_r_err, 2)[:])

    for c in 1:4
        println("color $(color_names[c]) err: ", 
            sum(color_err[:, :, c], 2)[:] / (N - num_na[c]))
    end
end


f = open(ARGS[2])
if ARGS[1] == "--report"
    for line in eachline(f)
        report_on_stamp(strip(line))
    end
elseif ARGS[1] == "--score"
    stamp_ids = [strip(line) for line in readlines(f)]
    score_stamps(stamp_ids)
end
close(f)

