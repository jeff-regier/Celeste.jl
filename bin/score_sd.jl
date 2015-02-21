#!/usr/bin/env julia

using DataFrames

include("score_cached.jl")

celeste_df = readtable("celeste.csv")
coadd_df = readtable("coadd.csv")
celeste_err = get_err_df(coadd_df, celeste_df)



### color UQ

for c in 1:4
    color_col = symbol("color_$(color_names[c])")
    color_sd_col = symbol("color_$(color_names[c])_sd")

    non_triv = celeste_df[color_sd_col] .>= 0.0002;

    @printf("%s & %.2f (%.2f) & %d & %.2f (%.2f) & %d \\\\\n",
        string(color_col),
        mean(celeste_err[!non_triv, color_col]),
        std(celeste_err[!non_triv, color_col]) / sqrt(sum(!non_triv)),
        sum(!non_triv),
        mean(celeste_err[non_triv, color_col]),
        std(celeste_err[non_triv, color_col]) / sqrt(sum(non_triv)),
        sum(non_triv))
end


### is_star UQ

non_triv = 0.011 .< celeste_df[:is_star] .< .989

wrong_in_non_triv = sum(celeste_err[non_triv, :missed_stars]) + 
        sum(celeste_err[non_triv, :missed_gals])
wrong_in_triv = sum(celeste_err[!non_triv, :missed_stars]) + 
        sum(celeste_err[!non_triv, :missed_gals])

@printf(" %d / %d    %d / %d\n",
    wrong_in_triv,
    sum(!non_triv),
    wrong_in_non_triv,
    sum(non_triv))


### brightness

println("r-r_err cor: ", cor(celeste_df[:flux_r], celeste_err[:flux_r]))
println("r_sd-r_err cor: ", cor(celeste_df[:flux_r_sd], celeste_err[:flux_r]))
