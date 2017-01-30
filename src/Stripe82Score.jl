module Stripe82Score

import JLD
import FITSIO
using DataFrames

import ..SDSSIO
import ..SDSSIO: RunCamcolField
import ..Model: CatalogEntry, ids
import ..ParallelRun: OptimizedSource


immutable MatchException <: Exception
    msg::String
end


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
load_s82(fname)

Load Stripe 82 objects into a DataFrame. `fname` should be a FITS file
created by running a CasJobs (skyserver.sdss.org/casjobs/) query
on the Stripe82 database. Run the following query in the \"Stripe82\"
context, then download the table as a FITS file.

```
select
  objid, rerun, run, camcol, field, flags,
  ra, dec, probpsf,
  psfmag_u, psfmag_g, psfmag_r, psfmag_i, psfmag_z,
  devmag_u, devmag_g, devmag_r, devmag_i, devmag_z,
  expmag_u, expmag_g, expmag_r, expmag_i, expmag_z,
  fracdev_r,
  devab_r, expab_r,
  devphi_r, expphi_r,
  devrad_r, exprad_r
into mydb.s82_0_1_0_1
from stripe82.photoobj
where
  run in (106, 206) and
  ra between 0. and 1. and
  dec between 0. and 1.
```
"""
function load_s82(fname)
    # First, simply read the FITS table into a dictionary of arrays.
    f = FITSIO.FITS(fname)
    keys = [:objid, :rerun, :run, :camcol, :field, :flags,
            :ra, :dec, :probpsf,
            :psfmag_u, :psfmag_g, :psfmag_r, :psfmag_i, :psfmag_z,
            :devmag_u, :devmag_g, :devmag_r, :devmag_i, :devmag_z,
            :expmag_u, :expmag_g, :expmag_r, :expmag_i, :expmag_z,
            :fracdev_r,
            :devab_r, :expab_r,
            :devphi_r, :expphi_r,
            :devrad_r, :exprad_r,
            :flags]
    objs = Dict(key=>read(f[2], string(key)) for key in keys)
    close(f)

    usedev = objs[:fracdev_r] .> 0.5  # true=> use dev, false=> use exp

    # Convert to "celeste" style results.
    gal_mag_u = where(usedev, objs[:devmag_u], objs[:expmag_u])
    gal_mag_g = where(usedev, objs[:devmag_g], objs[:expmag_g])
    gal_mag_r = where(usedev, objs[:devmag_r], objs[:expmag_r])
    gal_mag_i = where(usedev, objs[:devmag_i], objs[:expmag_i])
    gal_mag_z = where(usedev, objs[:devmag_z], objs[:expmag_z])

    result = DataFrame()
    result[:objid] = objs[:objid]
    result[:ra] = objs[:ra]
    result[:dec] = objs[:dec]
    result[:is_star] = [x != 0 for x in objs[:probpsf]]
    result[:star_mag_r] = objs[:psfmag_r]
    result[:gal_mag_r] = gal_mag_r

    # star colors
    result[:star_color_ug] = objs[:psfmag_u] .- objs[:psfmag_g]
    result[:star_color_gr] = objs[:psfmag_g] .- objs[:psfmag_r]
    result[:star_color_ri] = objs[:psfmag_r] .- objs[:psfmag_i]
    result[:star_color_iz] = objs[:psfmag_i] .- objs[:psfmag_z]

    # gal colors
    result[:gal_color_ug] = gal_mag_u .- gal_mag_g
    result[:gal_color_gr] = gal_mag_g .- gal_mag_r
    result[:gal_color_ri] = gal_mag_r .- gal_mag_i
    result[:gal_color_iz] = gal_mag_i .- gal_mag_z

    # gal shape -- fracdev
    result[:gal_fracdev] = objs[:fracdev_r]

    # Note that the SDSS photo pipeline doesn't constrain the de Vaucouleur
    # profile parameters and exponential disk parameters (A/B, angle, scale)
    # to be the same, whereas Celeste does. Here, we pick one or the other
    # from SDSS, based on fracdev - we'll get the parameters corresponding
    # to the dominant component. Later, we limit comparison to objects with
    # fracdev close to 0 or 1 to ensure that we're comparing apples to apples.

    result[:gal_ab] = where(usedev, objs[:devab_r], objs[:expab_r])

    # gal effective radius (re)
    re_arcsec = where(usedev, objs[:devrad_r], objs[:exprad_r])
    re_pixel = re_arcsec / 0.396
    result[:gal_scale] = re_pixel

    # gal angle (degrees)
    raw_phi = where(usedev, objs[:devphi_r], objs[:expphi_r])
    result[:gal_angle] = raw_phi - floor.(raw_phi / 180) * 180

    return result
end


"""
Convert two fluxes to a color: mag(f1) - mag(f2), assuming the same zeropoint.
Returns NaN if either flux is nonpositive.
"""
function fluxes_to_color(f1::Real, f2::Real)
    (f1 <= 0. || f2 <= 0.) && return NaN
    return -2.5 * log10(f1 / f2)
end


"""
Load the SDSS photoObj catalog used to initialize celeste, and reformat column
names to match what the rest of the scoring code expects.
"""
function load_primary(rcf::RunCamcolField, stagedir::String)
    dir = @sprintf "%s/%d/%d/%d" stagedir rcf.run rcf.camcol rcf.field
    fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir rcf.run rcf.camcol rcf.field
    println(fname)
    objs = SDSSIO.read_photoobj(fname)

    usedev = objs["frac_dev"] .> 0.5  # true=> use dev, false=> use exp

    gal_flux_u = where(usedev, objs["devflux_u"], objs["expflux_u"])
    gal_flux_g = where(usedev, objs["devflux_g"], objs["expflux_g"])
    gal_flux_r = where(usedev, objs["devflux_r"], objs["expflux_r"])
    gal_flux_i = where(usedev, objs["devflux_i"], objs["expflux_i"])
    gal_flux_z = where(usedev, objs["devflux_z"], objs["expflux_z"])

    result = DataFrame()
    result[:objid] = objs["objid"]
    result[:ra] = objs["ra"]
    result[:dec] = objs["dec"]
    result[:is_star] = objs["is_star"]
    result[:star_mag_r] = flux_to_mag.(objs["psfflux_r"])
    result[:gal_mag_r] = flux_to_mag.(gal_flux_r)

    # star colors
    result[:star_color_ug] = fluxes_to_color.(objs["psfflux_u"], objs["psfflux_g"])
    result[:star_color_gr] = fluxes_to_color.(objs["psfflux_g"], objs["psfflux_r"])
    result[:star_color_ri] = fluxes_to_color.(objs["psfflux_r"], objs["psfflux_i"])
    result[:star_color_iz] = fluxes_to_color.(objs["psfflux_i"], objs["psfflux_z"])

    # gal colors
    result[:gal_color_ug] = fluxes_to_color.(gal_flux_u, gal_flux_g)
    result[:gal_color_gr] = fluxes_to_color.(gal_flux_g, gal_flux_r)
    result[:gal_color_ri] = fluxes_to_color.(gal_flux_r, gal_flux_i)
    result[:gal_color_iz] = fluxes_to_color.(gal_flux_i, gal_flux_z)

    # gal shape -- fracdev
    result[:gal_fracdev] = objs["frac_dev"]

    # gal shape -- axis ratio
    #TODO: filter when 0.5 < frac_dev < .95
    result[:gal_ab] = where(usedev, objs["ab_dev"], objs["ab_exp"])

    # gal effective radius (re)
    re_arcsec = where(usedev, objs["theta_dev"], objs["theta_exp"])
    result[:gal_scale] = re_arcsec / 0.396 #pixel scale

    # gal angle (degrees)
    raw_phi = where(usedev, objs["phi_dev"], objs["phi_exp"])
    result[:gal_angle] = raw_phi - floor.(raw_phi / 180) * 180

    return result#[!objs["is_saturated"], :]
end


const color_names = ["ug", "gr", "ri", "iz"]

"""
This function loads one catalog entry into row of i of df, a results data
frame.
ce = Catalog Entry, a row of an astronomical catalog
"""
function load_ce!(i::Int, ce::CatalogEntry, df::DataFrame)
    df[i, :ra] = ce.pos[1]
    df[i, :dec] = ce.pos[2]
    df[i, :is_star] = ce.is_star ? 1. : 0.

    for j in 1:2
        s_type = ["star", "gal"][j]
        fluxes = j == 1 ? ce.star_fluxes : ce.gal_fluxes
        df[i, Symbol("$(s_type)_mag_r")] = flux_to_mag(fluxes[3])
        for c in 1:4
            cc = Symbol("$(s_type)_color_$(color_names[c])")
            cc_sd = Symbol("$(s_type)_color_$(color_names[c])_sd")
            if fluxes[c] > 0 && fluxes[c + 1] > 0  # leave as NA otherwise
                df[i, cc] = -2.5log10(fluxes[c] / fluxes[c + 1])
            else
                df[i, cc] = NaN
            end
        end
    end

    df[i, :gal_fracdev] = ce.gal_frac_dev
    df[i, :gal_ab] = ce.gal_ab
    df[i, :gal_angle] = (180/pi)ce.gal_angle
    df[i, :gal_scale] = ce.gal_scale * sqrt(ce.gal_ab)
    df[i, :objid] = ce.objid
end


"""
Convert Celeste results to a dataframe.
"""
function celeste_to_df(results::Vector{OptimizedSource})
    # Initialize dataframe
    N = length(results)
    color_col_names = ["color_$cn" for cn in color_names]
    color_sd_col_names = ["color_$(cn)_sd" for cn in color_names]
    col_names = vcat(["objid", "ra", "dec", "is_star", "star_mag_r",
                      "star_mag_r_sd", "gal_mag_r", "gal_mag_r_sd"],
                     ["star_$c" for c in color_col_names],
                     ["star_$c" for c in color_sd_col_names],
                     ["gal_$c" for c in color_col_names],
                     ["gal_$c" for c in color_sd_col_names],
                     ["gal_fracdev", "gal_ab", "gal_angle", "gal_scale"])
    col_Symbols = Symbol[Symbol(cn) for cn in col_names]
    col_types = Vector{DataType}(length(col_names))
    fill!(col_types, Float64)
    col_types[1] = String
    df = DataFrame(col_types, N)
    names!(df, col_Symbols)

    # Fill dataframe row-by-row.
    i = 0
    for result in results
        i += 1
        vs = result.vs

        function get_median_fluxes(i::Int)
            ret = Vector{Float64}(5)
            ret[3] = exp(vs[ids.r1[i]])
            ret[4] = ret[3] * exp(vs[ids.c1[3, i]])
            ret[5] = ret[4] * exp(vs[ids.c1[4, i]])
            ret[2] = ret[3] / exp(vs[ids.c1[2, i]])
            ret[1] = ret[2] / exp(vs[ids.c1[1, i]])
            ret
        end

        ce = CatalogEntry(
            vs[ids.u],
            vs[ids.a[1, 1]] > 0.5,
            get_median_fluxes(1),
            get_median_fluxes(2),
            vs[ids.e_dev],
            vs[ids.e_axis],
            vs[ids.e_angle],
            vs[ids.e_scale],
            result.objid,
            result.thingid)
        load_ce!(i, ce, df)

        df[i, :is_star] = vs[ids.a[1, 1]]

        #TODO: update UQ to mag units not flux. Also, log-normal now, not gamma.
#        for j in 1:2
#            s_type = ["star", "gal"][j]
#            df[i, Symbol("$(s_type)_flux_r_sd")] =
#                sqrt(df[i, Symbol("$(s_type)_flux_r")]) * vs[ids.r2[j]]
#            for c in 1:4
#                cc_sd = Symbol("$(s_type)_color_$(color_names[c])_sd")
#                df[i, cc_sd] = 2.5 * log10(e) * vs[ids.c2[c, j]]
#            end
#        end
    end

    return df
end


"""
Given two results data frame, one containing ground truth (i.e Coadd)
and one containing predictions (i.e., either Primary of Celeste),
compute an a data frame containing each prediction's error.
(It's not an average of the errors, it's each error.)
Let's call the return type of this function an \"error data frame\".
"""
function get_err_df(truth::DataFrame, predicted::DataFrame)
    color_cols = [Symbol("color_$cn") for cn in color_names]
    abs_err_cols = [:gal_fracdev, :gal_ab, :gal_scale]
    col_Symbols = vcat([:objid, :position, :missed_stars,
                        :missed_gals, :mag_r],
                       color_cols,
                       abs_err_cols,
                       [:gal_angle])

    col_types = fill(Float64, length(col_Symbols))
    col_types[1] = String
    col_types[3] = col_types[4] = Bool
    ret = DataFrame(col_types, size(truth, 1))
    names!(ret, col_Symbols)
    ret[:objid] = truth[:objid]

    predicted_gal = predicted[:is_star] .< .5
    true_gal = truth[:is_star] .< .5
    ret[:missed_stars] =  predicted_gal & !(true_gal)
    ret[:missed_gals] =  !predicted_gal & true_gal

    ret[:position] = dist.(truth[:ra], truth[:dec],
                          predicted[:ra], predicted[:dec])

    ret[true_gal, :mag_r] =
        abs(truth[true_gal, :gal_mag_r] - predicted[true_gal, :gal_mag_r])
    ret[!true_gal, :mag_r] =
        abs(truth[!true_gal, :star_mag_r] - predicted[!true_gal, :star_mag_r])

    for cn in color_names
        ret[true_gal, Symbol("color_$cn")] =
            abs(truth[true_gal, Symbol("gal_color_$cn")] -
                predicted[true_gal, Symbol("gal_color_$cn")])
        ret[!true_gal, Symbol("color_$cn")] =
            abs(truth[!true_gal, Symbol("star_color_$cn")] -
                predicted[!true_gal, Symbol("star_color_$cn")])
    end

    for n in abs_err_cols
        ret[n] = abs(predicted[n] - truth[n])
    end

    function degrees_to_diff(a, b)
        angle_between = abs(a - b) % 180
        min.(angle_between, 180 - angle_between)
    end

    ret[:gal_angle] = degrees_to_diff(truth[:gal_angle], predicted[:gal_angle])

    ret
end


function match_catalogs(rcf::RunCamcolField,
               results, truthfile, stagedir)
    # convert Celeste results to a DataFrame.
    celeste_full_df = celeste_to_df(results)
    println("celeste: $(size(celeste_full_df, 1)) objects")

    # load coadd catalog
    coadd_full_df = load_s82(truthfile)
    println("coadd catalog: $(size(coadd_full_df, 1)) objects")

    # load "primary" catalog (the SDSS photoObj catalog used to initialize
    # celeste).
    primary_full_df = load_primary(rcf, stagedir)
    println("primary catalog: $(size(primary_full_df, 1)) objects")

    # find matches in coadd catalog by position
    disttol = 1.0 / 0.396  # 1 arcsec

    good_celeste_indexes = Int[]
    good_coadd_indexes = Int[]
    good_primary_indexes = Int[]

    for i in 1:size(primary_full_df, 1)
        try
            j = match_position(coadd_full_df[:ra], coadd_full_df[:dec],
                         primary_full_df[i, :ra], primary_full_df[i, :dec],
                         disttol)
            # very large galaxies make average L1 error an unsuitable metric
            # for comparing scale, but other metrics are less interpretable,
            # so let's omit any galaxy that is more than an order of
            # magnitude larger than an average-size galaxy.
            if coadd_full_df[j, :gal_scale] > 20
                continue
            end

            # primary is better at flagging oversaturated sources that coadd
            if primary_full_df[i, :star_mag_r] < 16
                continue
            end

            k = findfirst(celeste_full_df[:objid], primary_full_df[i, :objid])
            # Celeste doesn't process some light sources due to various filters
            if k != 0
                push!(good_primary_indexes, i)
                push!(good_coadd_indexes, j)
                push!(good_celeste_indexes, k)
            end
        catch y
            isa(y, MatchException) || throw(y)
        end
    end

    primary_df = primary_full_df[good_primary_indexes, :]
    celeste_df = celeste_full_df[good_celeste_indexes, :]
    coadd_df = coadd_full_df[good_coadd_indexes, :]

    # show that all catalogs have same size, and (hopefully)
    # that not too many sources were filtered
    println("matched celeste catalog: $(size(celeste_df, 1)) objects")
    println("matched coadd catalog: $(size(coadd_df, 1)) objects")
    println("matched primary catalog: $(size(primary_df, 1)) objects")

    (celeste_df, primary_df, coadd_df)
end


function get_scores_df(celeste_err, primary_err, coadd_df)
    ttypes = [Symbol, Float64, Float64, Float64, Float64, Int]
    scores_df = DataFrame(ttypes, size(celeste_err, 2) - 1)
    names!(scores_df, [:field, :primary, :celeste, :diff, :diff_sd, :N])

    for i in 1:(size(celeste_err, 2) - 1)
        nm = names(celeste_err)[i + 1]
        nm != :objid || continue

        pe_good = Bool[!isnan(x) for x in primary_err[:, nm]]
        ce_good = Bool[!isnan(x) for x in celeste_err[:, nm]]
        good_row = pe_good & ce_good
        if string(nm)[1:5] == "star_"
            good_row &= (coadd_df[:is_star] .> 0.5)
        elseif string(nm)[1:4] == "gal_"
            good_row &= (coadd_df[:is_star] .< 0.5)
            if nm in [:gal_ab, :gal_scale, :gal_angle, :gal_fracdev]
                good_row &= !(0.05 .< coadd_df[:gal_fracdev] .< 0.95)
            end
            if nm == :gal_angle
                good_row &= coadd_df[:gal_ab] .< .6
            end
        end

        scores_df[i, :field] = nm
        N_good = sum(good_row)
        scores_df[i, :N] = N_good
        N_good > 0 || continue

        scores_df[i, :primary] = mean(primary_err[good_row, nm])
        scores_df[i, :celeste] = mean(celeste_err[good_row, nm])

        diffs = primary_err[good_row, nm] .- celeste_err[good_row, nm]
        scores_df[i, :diff] = mean(diffs)

        # compute the difference in error rates between celeste and primary
        # if we have enough data to get confidence intervals
        N_good > 1 || continue
        abs_errs = Float64[abs(x) for x in diffs]
        scores_df[i, :diff_sd] = std(abs_errs) / sqrt(N_good)
    end

    scores_df
end


function score_field(rcf::RunCamcolField, results, truthfile, stagedir, outfname)
    (celeste_df, primary_df, coadd_df) = match_catalogs(rcf,
                                results, truthfile, stagedir)

    # difference between celeste and coadd
    celeste_err = get_err_df(coadd_df, celeste_df)
    primary_err = get_err_df(coadd_df, primary_df)

    JLD.save(outfname,
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
function score_field_disk(rcf::RunCamcolField, fname, resultdir, truthfile, stagedir, outfname)
    results = JLD.load(fname, "results")
    println("Errors results (lower is better).  diff = Primary error - Celeste error.")
    println( score_field(rcf, results, truthfile, stagedir, outfname) )
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
