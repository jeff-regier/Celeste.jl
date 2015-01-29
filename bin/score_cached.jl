#!/usr/bin/env julia

using Celeste
using CelesteTypes

using DataFrames
import WCSLIB


function load_cache(stamp_id)
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


function score_cached(stamp_id)
    blob = SDSS.load_stamp_blob(ENV["STAMP"], stamp_id)
    true_cat_df = SDSS.load_stamp_catalog_df(ENV["STAMP"], "s82-$stamp_id", blob)
    true_cat = SDSS.load_stamp_catalog(ENV["STAMP"], "s82-$stamp_id", blob)
    baseline_cat_df = SDSS.load_stamp_catalog_df(ENV["STAMP"], stamp_id, blob,
         match_blob=true)
    baseline_cat = SDSS.load_stamp_catalog(ENV["STAMP"], stamp_id, blob,
         match_blob=true)
    mp = load_cache(stamp_id)

    println("================= STAMP $stamp_id ====================")

	vs = center_obj(mp.vp)
    true_ce = center_obj(true_cat)
    true_row = center_obj(true_cat_df)
    base_ce = center_obj(baseline_cat)
    base_row = center_obj(baseline_cat_df)

    print_comparison("position (pixel coordinates)", 
        true_ce.pos, round(base_ce.pos, 3), round(vs[ids.mu], 3))

    print_comparison("celestial body type",
        true_ce.is_star ? "star" : "galaxy", 
        base_ce.is_star ? "star" : "galaxy",
        vs[ids.chi] < .5 ? 
            "star ($(100 - 100vs[ids.chi])% certain)" :
            "galaxy ($(100vs[ids.chi])% certain)")

    color_names = ["$(band_letters[i+1])-$(band_letters[i])" for i in 1:4]

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


if length(ARGS) > 0
    score_cached(ARGS[1])
else
    for line in eachline(STDIN)
        score_cached(strip(line))
    end
end

