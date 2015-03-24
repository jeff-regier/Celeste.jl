#!/usr/bin/env julia

using DataFrames

include("score_cached.jl")

celeste_df = readtable("celeste.csv")
coadd_df = readtable("coadd.csv")
celeste_err = get_err_df(coadd_df, celeste_df)


### is_star UQ

non_triv = 0.011 .< celeste_df[:is_star] .< .989

wrong_in_non_triv = sum(celeste_err[non_triv, :missed_stars]) + 
        sum(celeste_err[non_triv, :missed_gals])
wrong_in_triv = sum(celeste_err[!non_triv, :missed_stars]) + 
        sum(celeste_err[!non_triv, :missed_gals])

@printf("uncertain: %d / %d\ncertain: %d / %d\n\n",
    wrong_in_non_triv,
    sum(non_triv),
    wrong_in_triv,
    sum(!non_triv))


### brightness

true_gal = coadd_df[:is_star] .< 0.5
flux_sd = celeste_df[:star_flux_r_sd]
flux_sd[true_gal] = celeste_df[true_gal, :gal_flux_r_sd]
qqs = quantile(flux_sd)

for i in 1:4
    pop = qqs[i] .<= flux_sd .<= qqs[i + 1]
    @printf("r_s quartile error: %.2f (%.2f)\n", 
        mean(celeste_err[pop, :flux_r]),
        std(celeste_err[pop, :flux_r]) / sqrt(sum(pop)))
end
println()


### color UQ

for c in 1:4
    color_col = symbol("color_$(color_names[c])")
    color_sd_col = symbol("star_color_$(color_names[c])_sd")

    color_sd = celeste_df[symbol("star_color_$(color_names[c])_sd")]
    color_sd[true_gal] = celeste_df[true_gal, symbol("gal_color_$(color_names[c])_sd")]

    qqs = quantile(color_sd)
    for i in 1:4
        pop = qqs[i] .<= color_sd .<= qqs[i + 1]
        @printf("c_s%d quartile %d error: %.2f (%.2f)\n",
            c, i,
            mean(celeste_err[pop, color_col]),
            std(celeste_err[pop, color_col]) / sqrt(sum(pop)))
    end
    println()
end

