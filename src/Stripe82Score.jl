module Stripe82Score

import JLD
import FITSIO
using DataFrames

import ..AccuracyBenchmark
import ..SDSSIO
import ..SDSSIO: RunCamcolField
import ..Model: CatalogEntry, ids
import ..ParallelRun: OptimizedSource




"""
mag_to_flux(m)

convert SDSS mags to SDSS flux
"""
mag_to_flux(m::AbstractFloat) = 10.^(0.4 * (22.5 - m))

flux_to_mag(nm::AbstractFloat) = nm > 0 ? 22.5 - 2.5 * log10(nm) : NaN

"""
where(condition, x, y)

Construct a new Array containing elements from `x` where `condition` is true
otherwise elements from `y`.
"""
function where(condition, x, y)
    @assert length(condition) == length(x) == length(y)
    out = similar(x)
    for i=1:length(condition)
        out[i] = condition[i]? x[i]: y[i]
    end
    return out
end


"""
Return distance in pixels using small-distance approximation. Falls
apart at poles and RA boundary.
"""
dist(ra1, dec1, ra2, dec2) = (3600 / 0.396) * (sqrt((dec2 - dec1).^2 +
                                  (cos(dec1) .* (ra2 - ra1)).^2))

"""
match_position(ras, decs, ra, dec, dist)

Return index of first position in `ras`, `decs` that is within a distance
`maxdist` (in degrees) of the target position `ra`, `dec`. If none found,
an exception is raised.
"""
function match_position(ras, decs, ra, dec, maxdist)
    @assert length(ras) == length(decs)
    for i in 1:length(ras)
        dist(ra, dec, ras[i], decs[i]) < maxdist && return i
    end
    throw(MatchException(@sprintf("No source found at %f  %f", ra, dec)))
end


"""
Convert two fluxes to a color: mag(f1) - mag(f2), assuming the same zeropoint.
Returns NaN if either flux is nonpositive.
"""
function fluxes_to_color(f1::Real, f2::Real)
    (f1 <= 0. || f2 <= 0.) && return NaN
    return -2.5 * log10(f1 / f2)
end

const color_names = ["ug", "gr", "ri", "iz"]







function score_field(rcf::RunCamcolField, results, truthfile, stagedir)
    (celeste_df, primary_df, coadd_df) = match_catalogs(rcf,
                                results, truthfile, stagedir)

    # difference between celeste and coadd
    celeste_err = get_err_df(coadd_df, celeste_df)
    primary_err = get_err_df(coadd_df, primary_df)

    JLD.save("results_and_errors.jld",
             "celeste_df", celeste_df,
             "primary_df", primary_df,
             "coadd_df", coadd_df,
             "celeste_err", celeste_err,
             "primary_err", primary_err)

    # create scores
    get_scores_df(celeste_err, primary_err, coadd_df)
end


"""
Score the celeste results for a particular field
"""
function score_field_disk(rcf::RunCamcolField, resultdir, truthfile, stagedir)
    fname = @sprintf "%s/celeste-%06d-%d-%04d.jld" resultdir rcf.run rcf.camcol rcf.field
    results = JLD.load(fname, "results")

    println( score_field(rcf, results, truthfile, stagedir) )
end


"""
Display results from Celeste, Primary, and Coadd for a particular object
"""
function score_object_disk(rcf::RunCamcolField, objid, resultdir, truthfile, stagedir)
    fname = @sprintf "%s/celeste-objid-%s.jld" resultdir objid
    results = JLD.load(fname, "results")

    (celeste_df, primary_df, coadd_df) = match_catalogs(rcf,
                                results, truthfile, stagedir)

    println("\n\nceleste results:\n")
    println(celeste_df)
    println("\n\nprimary results:\n")
    println(primary_df)
    println("\n\ncoadd results:\n")
    println(coadd_df)
end


end
