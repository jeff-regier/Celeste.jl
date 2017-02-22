#!/usr/bin/env julia

using FITSIO
using Celeste
using JLD
using DataFrames


function optimized_sources_to_fits(inpath, outdir)
    optimized_sources = load(inpath)["results"]
    sources_df = Celeste.AccuracyBenchmark.celeste_to_df(optimized_sources)

    # stores the optimized sources in an "attribute-major" format
    # suitable for writing to a fits table
    sources_dict = Dict{String, Vector}()
    sources_dict["thingid"] = [os.thingid for os in optimized_sources]
    for col_symbol in names(sources_df)
        if !any(isna, sources_df[:, col_symbol]) && col_symbol != :objid
            sources_dict[string(col_symbol)] = sources_df[col_symbol]
        end
    end

    outpath = joinpath(outdir, "$(basename(inpath)[1:end-4]).fits")
    outfits = FITS(outpath, "w")
    write(outfits, sources_dict)
    println("wrote $(length(optimized_sources)) to $outpath")
end


if length(ARGS) != 2 || ARGS[1][end-3:end] != ".jld"
    println("usage: optimized_sources_to_fits.jl <celeste-*.jld> <outdir>")
else
    optimized_sources_to_fits(ARGS[1], ARGS[2])
end
