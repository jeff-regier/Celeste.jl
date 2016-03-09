# Functions for interacting with Celeste from the command line.

using DataFrames
import FITSIO
import JLD
import SloanDigitalSkySurvey: SDSS

using .Types
import .SkyImages
import .ModelInit
import .OptimizeElbo

const TILE_WIDTH = 20
const MAX_ITERS = 50
const MIN_FLUX = 10


"""
Fit the Celeste model to a set of sources and write the output to a JLD file.

Args:
  - dir: The directory containing the FITS files.
  - run: An ASCIIString with the six-digit run number, e.g. "003900"
  - camcol: An ASCIIString with the camcol, e.g. "6"
  - field: An ASCIIString with the four-digit field, e.g. "0269"
  - outdir: The directory to write the output jld file.
  - partnum: Which of the 1:parts catalog entries to fit.
  - parts: How many parts to divide the catalog entries into

Returns:
  Writes a jld file to outdir containing the optimization output.
"""
function infer(
      dir::AbstractString, run::AbstractString, camcol::AbstractString,
      field::AbstractString, outdir::AbstractString,
      partnum::Int, parts::Int)

    # get images
    images = SkyImages.load_sdss_blob(dir, run, camcol, field)

    # load catalog and convert to Array of `CatalogEntry`s.
    cat_df = SDSS.load_catalog_df(dir, run, camcol, field)
    # TODO: fliter bad objects
    cat_entries = SkyImages.convert_catalog_to_celeste(cat_df, images)

    # limit to just the part of the catalog specified.
    partsize = length(cat_entries) / parts
    minidx = round(Int, partsize * (partnum - 1)) + 1
    maxidx = round(Int, partsize * partnum)
    cat_entries = cat_entries[minidx:maxidx]

    # initialize tiled images and model parameters
    tiled_blob, mp = ModelInit.initialize_celeste(images, cat_entries,
                                                  tile_width=TILE_WIDTH,
                                                  fit_psf=true)

    # Initialize output dictionary
    nsources = length(minidx:maxidx)
    out = Dict("obj" => minidx:maxidx,  # index within field
               "objid" => Array(ASCIIString, nsources),
               "vp" => Array(Vector{Float64}, nsources),
               "fit_time"=> Array(Float64, nsources))

    # Loop over sources in model
    for i in 1:mp.S
        println("Processing source $i, objid $(mp.objids[i])")

        mp_s = deepcopy(mp);
        mp_s.active_sources = [i]

        # TODO: This is slow but would run much faster if you had run
        # limit_to_object_data() first.
        trimmed_tiled_blob = ModelInit.trim_source_tiles(i, mp_s, tiled_blob;
                                                         noise_fraction=0.1);

        fit_time = time()
        iter_count, max_f, max_x, result =
            OptimizeElbo.maximize_f(ElboDeriv.elbo, trimmed_tiled_blob, mp_s;
                                    verbose=true, max_iters=MAX_ITERS)
        fit_time = time() - fit_time

        out["objid"][i] = mp_s.objids[i]
        out["vp"][i] = mp_s.vp[i]
        out["fit_time"][i] = fit_time
    end

    outfile = "$outdir/celeste-$run-$camcol-$field--part-$partnum-$parts.jld"
    JLD.save(outfile, out)
end


"""
mag_to_flux(m)

convert SDSS mags to SDSS flux
"""
mag_to_flux(m::AbstractFloat) = 10.^(0.4 * (22.5 - m))
@vectorize_1arg AbstractFloat mag_to_flux

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
Return distance in degrees using small-distance approximation. Falls
apart at poles and RA boundary.
"""
dist(ra1, dec1, ra2, dec2) = sqrt((dec2 - dec1).^2 +
                                  (cos(dec1) .* (ra2 - ra1)).^2)

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
    error("No matching position found")
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
    objs = [key=>read(f[2], string(key)) for key in keys]
    close(f)

    # Convert to "celeste" style results.
    # Note that the SDSS photo pipeline doesn't constrain the de Vaucouleur
    # profile parameters and exponential disk parameters (A/B, angle, scale)
    # to be the same, whereas Celeste does. Here, we pick one or the other
    # from SDSS, based on fracdev - we'll get the parameters corresponding
    # to the dominant component. Later, we limit comparison to objects with
    # fracdev close to 0 or 1 to ensure that we're comparing apples to apples.
    usedev = objs[:fracdev_r] .> 0.5  # true=> use dev, false=> use exp
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
    result[:star_flux_r] = mag_to_flux(objs[:psfmag_r])
    result[:gal_flux_r] = mag_to_flux(gal_mag_r)

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

    # gal shape
    result[:gal_fracdev] = objs[:fracdev_r]
    result[:gal_ab] = where(usedev, objs[:devab_r], objs[:expab_r])
    result[:gal_angle] = where(usedev, objs[:devphi_r], objs[:expphi_r])
    result[:gal_scale] = where(usedev, objs[:devrad_r], objs[:exprad_r])

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
@vectorize_2arg Real fluxes_to_color


"""
load_primary(dir, run, camcol, field)

Load the SDSS photObj catalog used to initialize celeste, and reformat column
names to match what the rest of the scoring code expects.
"""
function load_primary(dir, run, camcol, field)

    objs = SDSS.load_catalog_df(dir, run, camcol, field)

    usedev = objs[:frac_dev] .> 0.5  # true=> use dev, false=> use exp
    gal_flux_u = where(usedev, objs[:devflux_u], objs[:expflux_u])
    gal_flux_g = where(usedev, objs[:devflux_g], objs[:expflux_g])
    gal_flux_r = where(usedev, objs[:devflux_r], objs[:expflux_r])
    gal_flux_i = where(usedev, objs[:devflux_i], objs[:expflux_i])
    gal_flux_z = where(usedev, objs[:devflux_z], objs[:expflux_z])


    result = DataFrame()
    result[:objid] = objs[:objid]
    result[:ra] = objs[:ra]
    result[:dec] = objs[:dec]
    result[:is_star] = objs[:is_star]
    result[:star_flux_r] = objs[:psfflux_r]
    result[:gal_flux_r] = gal_flux_r

    # star colors
    result[:star_color_ug] = fluxes_to_color(objs[:psfflux_u], objs[:psfflux_g])
    result[:star_color_gr] = fluxes_to_color(objs[:psfflux_g], objs[:psfflux_r])
    result[:star_color_ri] = fluxes_to_color(objs[:psfflux_r], objs[:psfflux_i])
    result[:star_color_iz] = fluxes_to_color(objs[:psfflux_i], objs[:psfflux_z])

    # gal colors
    result[:gal_color_ug] = fluxes_to_color(gal_flux_u, gal_flux_g)
    result[:gal_color_gr] = fluxes_to_color(gal_flux_g, gal_flux_r)
    result[:gal_color_ri] = fluxes_to_color(gal_flux_r, gal_flux_i)
    result[:gal_color_iz] = fluxes_to_color(gal_flux_i, gal_flux_z)

    # gal shape
    result[:gal_fracdev] = objs[:frac_dev]
    result[:gal_ab] = where(usedev, objs[:ab_dev], objs[:ab_exp])

    # TODO: the catalog contains both theta_[exp,dev] and phi_[exp,dev]!
    # which do we use?
    #result[:gal_angle] = where(usedev, objs[:theta_dev], objs[:theta_exp])
    result[:gal_angle] = fill(NaN, size(objs, 1))

    # TODO: No scale parameters in catalog.
    #result[:gal_scale] = where(usedev, objs[?], objs[?])
    result[:gal_scale] = fill(NaN, size(objs, 1))

    return result
end


"""
This function converts the parameters from Celeste for one light source
to a CatalogEntry. (which can be passed to load_ce!)
It only needs to be called by load_celeste_obj!
"""
function convert(::Type{CatalogEntry}, vs::Vector{Float64}, objid)
    function get_fluxes(i::Int)
        ret = Array(Float64, 5)
        ret[3] = exp(vs[ids.r1[i]] + 0.5 * vs[ids.r2[i]])
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
        vs[ids.e_scale],
        objid)
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
    df[i, :objid] = ce.objid
end



"""
Convert Celeste results to a dataframe.
"""
function celeste_to_df(vp::Vector{Vector{Float64}},
                       objid::Vector{ASCIIString})
    @assert length(vp) == length(objid)

    # Initialize dataframe
    N = length(vp)
    color_col_names = ["color_$cn" for cn in color_names]
    color_sd_col_names = ["color_$(cn)_sd" for cn in color_names]
    col_names = vcat(["objid", "ra", "dec", "is_star", "star_flux_r",
                      "star_flux_r_sd", "gal_flux_r", "gal_flux_r_sd"],
                     ["star_$c" for c in color_col_names],
                     ["star_$c" for c in color_sd_col_names],
                     ["gal_$c" for c in color_col_names],
                     ["gal_$c" for c in color_sd_col_names],
                     ["gal_fracdev", "gal_ab", "gal_angle", "gal_scale"])
    col_symbols = Symbol[symbol(cn) for cn in col_names]
    col_types = Array(DataType, length(col_names))
    fill!(col_types, Float64)
    col_types[1] = ASCIIString
    df = DataFrame(col_types, N)
    names!(df, col_symbols)

    # Fill dataframe row-by-row.
    for i=1:N
        ce = convert(CatalogEntry, vp[i], objid[i])
        load_ce!(i, ce, df)

        df[i, :is_star] = vp[i][ids.a[1]]

        for j in 1:2
            s_type = ["star", "gal"][j]
            df[i, symbol("$(s_type)_flux_r_sd")] =
                sqrt(df[i, symbol("$(s_type)_flux_r")]) * vp[i][ids.r2[j]]
            for c in 1:4
                cc_sd = symbol("$(s_type)_color_$(color_names[c])_sd")
                df[i, cc_sd] = 2.5 * log10(e) * vp[i][ids.c2[c, j]]
            end
        end
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
    color_cols = [symbol("color_$cn") for cn in color_names]
    abs_err_cols = [:gal_fracdev, :gal_ab, :gal_scale]
    col_symbols = vcat([:objid, :position, :missed_stars,
                        :missed_gals, :flux_r],
                       color_cols,
                       abs_err_cols,
                       :gal_angle)

    col_types = fill(Float64, length(col_symbols))
    col_types[1] = ASCIIString
    col_types[3] = col_types[4] = Bool
    ret = DataFrame(col_types, size(truth, 1))
    names!(ret, col_symbols)
    ret[:objid] = truth[:objid]

    predicted_gal = predicted[:is_star] .< .5
    true_gal = truth[:is_star] .< .5
    ret[:missed_stars] =  predicted_gal & !(true_gal)
    ret[:missed_gals] =  !predicted_gal & true_gal

    ret[:position] = dist(truth[:ra], truth[:dec],
                          predicted[:ra], predicted[:dec])

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

    function degrees_to_diff(a, b)
        angle_between = abs(a - b) % 180
        min(angle_between, 180 - angle_between)
    end

    ret[:gal_angle] = degrees_to_diff(truth[:gal_angle], predicted[:gal_angle])

    ret
end

"""
Score all the celeste results for sources in the given
(`run`, `camcol`, `field`).
This is done by finding all files with names matching
`DIR/celeste-RUN-CAMCOL-FIELD-*.jld`
"""
function score_field(dir, run, camcol, field, outdir, reffile)

    # find celeste result files matching the pattern
    re = Regex("celeste-$run-$camcol-$field-.*\.jld")
    fnames = filter(x->ismatch(re, x), readdir(outdir))
    paths = [joinpath(outdir, name) for name in fnames]

    # collect Celeste results for the field
    objid = ASCIIString[]
    vp = Vector{Float64}[]
    for path in paths
        d = load(path)
        append!(objid, d["objid"])
        append!(vp, d["vp"])
    end

    # convert Celeste results to a DataFrame.
    celeste_df = celeste_to_df(vp, objid)
    println("celeste: $(size(celeste_df, 1)) objects")

    # load coadd catalog
    coadd_full_df = load_s82(reffile)
    println("coadd catalog: $(size(coadd_full_df, 1)) objects")

    # find matches in coadd catalog by position
    disttol = 1.0 / 3600.0  # 1 arcsec
    matchidx = Int[match_position(coadd_full_df[:ra], coadd_full_df[:dec],
                                  celeste_df[i, :ra], celeste_df[i, :dec],
                                  disttol)
                   for i=1:size(celeste_df, 1)]

    # limit coadd to matched objects
    coadd_df = coadd_full_df[matchidx, :]

    # load "primary" catalog (the SDSS photObj catalog used to initialize
    # celeste).
    primary_full_df = load_primary(dir, run, camcol, field)
    println("primary catalog: $(size(primary_full_df, 1)) objects")

    # match by object id
    matchidx = Int[findfirst(primary_full_df[:objid], objid)
                   for objid in celeste_df[:objid]]

    # ensure that all objects are matched
    if countnz(matchidx) != size(celeste_df, 1)
        error("catalog mismatch between celeste and primary")
    end

    # limit primary to matched items
    primary_df = primary_full_df[matchidx, :]

    # difference between celeste and coadd
    celeste_err = get_err_df(coadd_df, celeste_df)
    primary_err = get_err_df(coadd_df, primary_df)

    # create scores
    ttypes = [Symbol, Float64, Float64, Float64, Float64, Int]
    scores_df = DataFrame(ttypes, size(celeste_err, 2) - 1)
    names!(scores_df, [:field, :primary, :celeste, :diff, :diff_sd, :N])
    for i in 1:(size(celeste_err, 2) - 1)
        n = names(celeste_err)[i + 1]
        if n == :objid
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

    scores_df
end
