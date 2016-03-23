# Functions for interacting with Celeste from the command line.

using DataFrames
import FITSIO
import JLD
using Logging  # just for testing right now

using .Types
import .SDSS
import .SkyImages
import .ModelInit
import .OptimizeElbo

const TILE_WIDTH = 20
const DEFAULT_MAX_ITERS = 50
const MIN_FLUX = 2.0


function set_logging_level(level)
    if level == "OFF"
      Logging.configure(level=OFF)
    elseif level == "DEBUG"
      Logging.configure(level=DEBUG)
    elseif level == "INFO"
      Logging.configure(level=INFO)
    elseif level == "WARNING"
      Logging.configure(level=WARNING)
    elseif level == "ERROR"
      Logging.configure(level=ERROR)
    elseif level == "CRITICAL"
      Logging.configure(level=CRITICAL)
    else
      err("Unknown logging level $(level)")
    end
end


immutable MatchException <: Exception
    msg::ASCIIString
end


"""
read_photoobj_primary(fieldids, dirs) -> Vector{CatalogEntry}

Combine photoobj catalogs for the given overlapping fields, returning a single
joined catalog containing only primary objects.
"""
function read_photoobj_primary(fieldids::Vector{Tuple{Int, Int, Int}}, dirs;
    ignore_primary_mask=false)
    @assert length(fieldids) == length(dirs)

    # if we're treating any detection as primary, limit processing to one field
    @assert !ignore_primary_mask || length(fieldids) == 1

    info("reading photoobj catalogs for ", length(fieldids), " fields")

    # the code below assumes there is at least one field.
    if length(fieldids) == 0
        return CatalogEntry[]
    end

    # Read in all photoobj catalogs.
    rawcatalogs = Array(Dict, length(fieldids))
    for i in eachindex(fieldids)
        run, camcol, field = fieldids[i]
        dir = dirs[i]
        fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir run camcol field
        info("field $(fieldids[i]): reading $fname")
        rawcatalogs[i] = SDSSIO.read_photoobj(fname)
    end

    for i in eachindex(fieldids)
        info("field ", fieldids[i], ": ", length(rawcatalogs[i]["objid"]),
             " entries")
    end

    # Limit each catalog to primary objects and objects where thing_id != -1
    # (thing_id == -1 indicates that the matching process failed)
    for cat in rawcatalogs
        mask = (cat["thing_id"] .!= -1)
        if !ignore_primary_mask
            mask &= (cat["mode"] .== 0x01)
        end
        for key in keys(cat)
            cat[key] = cat[key][mask]
        end
    end

    for i in eachindex(fieldids)
        info("field ", fieldids[i], ": ", length(rawcatalogs[i]["objid"]),
             " primary entries")
    end

    # Merge all catalogs together (there should be no duplicate objects,
    # because for each object there should only be one "primary" occurance.)
    rawcatalog = deepcopy(rawcatalogs[1])
    for i=2:length(rawcatalogs)
        for key in keys(rawcatalog)
            append!(rawcatalog[key], rawcatalogs[i][key])
        end
    end

    # check that there are no duplicate thing_ids (see above comment)
    if length(Set(rawcatalog["thing_id"])) < length(rawcatalog["thing_id"])
        error("Found one or more duplicate primary thing_ids in photoobj " *
              "catalogs")
    end

    # convert to celeste format catalog
    catalog = SkyImages.convert(Vector{CatalogEntry}, rawcatalog)

    return catalog
end


"""
infer(ra_range, dec_range, fieldids, frame_dirs; ...)

Fit the Celeste model to sources in a given ra, dec range.

- ra_range: minimum and maximum RA of sources to consider.
- dec_range: minimum and maximum Dec of sources to consider.
- fieldids: Array of run, camcol, field triplets that the source occurs in.
- frame_dirs: Directories in which to find each field's frame FITS files.

Returns:

- Dictionary of results, keyed by SDSS thing_id.
"""
function infer(ra_range::Tuple{Float64, Float64},
               dec_range::Tuple{Float64, Float64},
               fieldids::Vector{Tuple{Int, Int, Int}},
               frame_dirs::Vector;
               fpm_dirs=frame_dirs,
               psfield_dirs=frame_dirs,
               photofield_dirs=frame_dirs,
               photoobj_dirs=frame_dirs,
               max_iters=DEFAULT_MAX_ITERS,
               ignore_primary_mask=false)
    # Read all primary objects in these fields.
    catalog = read_photoobj_primary(fieldids, photoobj_dirs,
                      ignore_primary_mask=ignore_primary_mask)
    info("$(length(catalog)) primary sources")

    # Filter out low-flux objects in the catalog.
    catalog = filter(entry->(maximum(entry.star_fluxes) >= MIN_FLUX), catalog)
    info("$(length(catalog)) primary sources after MIN_FLUX cut")

    # Get indicies of entries in the  RA/Dec range of interest.
    entry_in_range = entry->((ra_range[1] < entry.pos[1] < ra_range[2]) &&
                             (dec_range[1] < entry.pos[2] < dec_range[2]))
    target_sources = find(entry_in_range, catalog)

    info("processing $(length(target_sources)) sources in RA/Dec range")

    # If there are no objects of interest, return early.
    if length(target_sources) == 0
        return Dict{Int, Dict}()
    end

    # Read in images for all (run, camcol, field).
    images = Image[]
    image_names = ASCIIString[]
    image_count = 0
    for i in 1:length(fieldids)
        info("reading field ", fieldids[i])
        run, camcol, field = fieldids[i]
        fieldims = SkyImages.read_sdss_field(run, camcol, field, frame_dirs[i],
                                             fpm_dir=fpm_dirs[i],
                                             psfield_dir=psfield_dirs[i],
                                             photofield_dir=photofield_dirs[i])
        for b=1:length(fieldims)
          image_count += 1
          push!(image_names,
                "$image_count run=$run camcol=$camcol $field=field b=$b")
        end
        append!(images, fieldims)
    end

    debug("Image names:")
    debug(image_names)

    # initialize tiled images and model parameters for trimming.  We will
    # initialize the psf again before fitting, so we don't do it here.
    info("initializing celeste without PSF fit")
    tiled_images = SkyImages.break_blob_into_tiles(images, TILE_WIDTH)
    mp = ModelInit.initialize_model_params(tiled_images, images, catalog,
                                           fit_psf=false)


    # get indicies of all sources relevant to those we're actually
    # interested in, and fit a local PSF for those sources (since we skipped
    # fitting the PSF for the whole catalog above)
    info("fitting PSF for all relevant sources")
    ModelInit.fit_object_psfs!(mp, target_sources, images)

    results = Dict{Int, Dict}()
    for s in target_sources
        entry = catalog[s]
        mp.active_sources = [s]

        info("processing source $s: objid= $(entry.objid)")

        t0 = time()
        trimmed_tiled_images = ModelInit.trim_source_tiles(s, mp, tiled_images;
                                                           noise_fraction=0.1)
        init_time = time() - t0

        t0 = time()
        iter_count, max_f, max_x, result =
            OptimizeElbo.maximize_f(ElboDeriv.elbo, trimmed_tiled_images, mp;
                                    verbose=true, max_iters=max_iters)
        fit_time = time() - t0

        results[entry.thing_id] = Dict("objid"=>entry.objid,
                                       "ra"=>entry.pos[1],
                                       "dec"=>entry.pos[2],
                                       "vs"=>mp.vp[s],
                                       "init_time"=>init_time,
                                       "fit_time"=>fit_time)
    end

    return results
end


"""
Infer a single objid in a single run, camcol, and field.
"""
function infer(
    run::Int, camcol::Int, field::Int, objid::AbstractString,
    dir::AbstractString)

  images = SkyImages.read_sdss_field(run, camcol, field, dir);

  cat_filename = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir run camcol field
  cat_entries = SkyImages.read_photoobj_celeste(joinpath(dir, cat_filename));

  # initialize tiled images and model parameters.  Don't fit the psf for now --
  # we just need the tile_sources from mp.
  tiled_blob, mp = ModelInit.initialize_celeste(images, cat_entries,
                                                tile_width=20,
                                                fit_psf=false);
  s = findfirst(mp.objids, objid)
  @assert(s > 0, "Objid $objid not found in the catalog.")
  relevant_sources = ModelInit.get_relevant_sources(mp, s);
  ModelInit.fit_object_psfs!(mp, relevant_sources, images);
  mp.active_sources = [ s ];

  #for objid in bad_objids
  trimmed_tiled_blob =
    ModelInit.trim_source_tiles(s, mp, tiled_blob, noise_fraction=0.1);

  fit_time = time()
  iter_count, max_f, max_x, result =
      OptimizeElbo.maximize_f(ElboDeriv.elbo, trimmed_tiled_blob, mp;
                              verbose=true, max_iters=50)
  fit_time = time() - fit_time

  info("Fit in $fit_time seconds.")
end


# -----------------------------------------------------------------------------
# NERSC-specific functions

"""
Query the SDSS database for all fields that overlap the given RA, Dec range.
"""
function query_overlapping_fields(ramin, ramax, decmin, decmax)

    # The following file was generated by running the query
    # """
    # select run, camcol, field, ramin, ramax, decmin, decmax
    # into mydb.field_extents from dr12.field
    # """
    # on CasJobs and then downloading the resulting table as FITS.
    fname = "/project/projectdirs/dasrepo/celeste-sc16/field_extents.fits"

    f = FITSIO.FITS(fname)
    hdu = f[2]::FITSIO.TableHDU

    # read in the entire table.
    all_run = read(hdu, "run")::Vector{Int16}
    all_camcol = read(hdu, "camcol")::Vector{UInt8}
    all_field = read(hdu, "field")::Vector{Int16}
    all_ramin = read(hdu, "ramin")::Vector{Float64}
    all_ramax = read(hdu, "ramax")::Vector{Float64}
    all_decmin = read(hdu, "decmin")::Vector{Float64}
    all_decmax = read(hdu, "decmax")::Vector{Float64}

    close(f)

    # initialize output "table"
    out = Dict{ASCIIString, Vector}("run"=>Int[],
                                    "camcol"=>Int[],
                                    "field"=>Int[],
                                    "ramin"=>Float64[],
                                    "ramax"=>Float64[],
                                    "decmin"=>Float64[],
                                    "decmax"=>Float64[])

    # The ramin, ramax, etc is a bit unintuitive because we're looking
    # for any overlap.
    for i in eachindex(all_ramin)
        if (all_ramax[i] > ramin && all_ramin[i] < ramax &&
            all_decmax[i] > decmin && all_decmin[i] < decmax)
            push!(out["run"], all_run[i])
            push!(out["camcol"], all_camcol[i])
            push!(out["field"], all_field[i])
            push!(out["ramin"], all_ramin[i])
            push!(out["ramax"], all_ramax[i])
            push!(out["decmin"], all_decmin[i])
            push!(out["decmax"], all_decmax[i])
        end
    end

    return out
end

"""
query_overlapping_fieldids(ramin, ramax, decmin, decmax) -> (Int, Int, Int)

Like `query_overlapping_fields`, but return a Vector of
(run, camcol, field) triplets.
"""
function query_overlapping_fieldids(ramin, ramax, decmin, decmax)
    fields = query_overlapping_fields(ramin, ramax, decmin, decmax)
    return Tuple{Int, Int, Int}[(fields["run"][i],
                                 fields["camcol"][i],
                                 fields["field"][i])
                                for i in eachindex(fields["run"])]
end



const NERSC_DATA_ROOT = "/global/projecta/projectdirs/sdss/data/sdss/dr12/boss"
nersc_photoobj_dir(run::Integer, camcol::Integer) =
    "$(NERSC_DATA_ROOT)/photoObj/301/$(run)/$(camcol)"
nersc_psfield_dir(run::Integer, camcol::Integer) =
    "$(NERSC_DATA_ROOT)/photo/redux/301/$(run)/objcs/$(camcol)"
nersc_photofield_dir(run::Integer) = "$(NERSC_DATA_ROOT)/photoObj/301/$(run)"

"""
nersc_frame_dir(run, camcol, field)

Uncompress the frame files to user's scratch and return the directory on
scratch containing the uncompressed files.
"""
function nersc_frame_dir(run::Integer, camcol::Integer, field::Integer)
    # Uncompress the frame (bz2) files to scratch
    srcdir = "$(NERSC_DATA_ROOT)/photoObj/frames/301/$(run)/$(camcol)"
    dstdir = joinpath(ENV["SCRATCH"], "celeste", "frames", "$(run)-$(camcol)")
    isdir(dstdir) || mkpath(dstdir)
    for band in ['u', 'g', 'r', 'i', 'z']
        srcfile = @sprintf("%s/frame-%s-%06d-%d-%04d.fits.bz2",
                           srcdir, band, run, camcol, field)
        dstfile = @sprintf("%s/frame-%s-%06d-%d-%04d.fits",
                           dstdir, band, run, camcol, field)
        if !isfile(dstfile)
            println("bzcat --keep $srcfile > $dstfile")
            Base.run(pipeline(`bzcat --keep $srcfile`, stdout=dstfile))
        end
    end
    return dstdir
end


"""
nersc_fpm_dir(run, camcol, field)

Uncompress the fpM files to user's scratch and return the directory on
scratch containing the uncompressed files.
"""
function nersc_fpm_dir(run::Integer, camcol::Integer, field::Integer)
    # It isn't strictly necessary to uncompress these, because FITSIO can handle
    # gzipped files. However, the celeste code assumes the filename ends with
    # ".fit", so we have to at least symlink the files to a new name.
    srcdir = "$(NERSC_DATA_ROOT)/photo/redux/301/$(run)/objcs/$(camcol)"
    dstdir = joinpath(ENV["SCRATCH"], "celeste", "fpm", "$(run)-$(camcol)")
    isdir(dstdir) || mkpath(dstdir)
    for band in ['u', 'g', 'r', 'i', 'z']
        srcfile = @sprintf("%s/fpM-%06d-%s%d-%04d.fit.gz",
                           srcdir, run, band, camcol, field)
        dstfile = @sprintf("%s/fpM-%06d-%s%d-%04d.fit",
                           dstdir, run, band, camcol, field)
        if !isfile(dstfile)
            println("gunzip --stdout $srcfile > $dstfile")
            Base.run(pipeline(`gunzip --stdout $srcfile`, stdout=dstfile))
        end
    end
    return dstdir
end


"""
NERSC-specific infer function, called from main entry point.
"""
function infer_nersc(ramin, ramax, decmin, decmax, outdir)
    # Get vector of (run, camcol, field) triplets overlapping this patch
    fieldids = query_overlapping_fieldids(ramin, ramax, decmin, decmax)

    # Get relevant directories corresponding to each field.
    frame_dirs = [nersc_frame_dir(x[1], x[2], x[3]) for x in fieldids]
    fpm_dirs = [nersc_fpm_dir(x[1], x[2], x[3]) for x in fieldids]
    psfield_dirs = [nersc_psfield_dir(x[1], x[2]) for x in fieldids]
    photoobj_dirs = [nersc_photoobj_dir(x[1], x[2]) for x in fieldids]
    photofield_dirs = [nersc_photofield_dir(x[1]) for x in fieldids]

    results = infer((ramin, ramax), (decmin, decmax), fieldids,
                    frame_dirs;
                    fpm_dirs=fpm_dirs,
                    psfield_dirs=psfield_dirs,
                    photofield_dirs=photofield_dirs,
                    photoobj_dirs=photoobj_dirs)

    fname = @sprintf("%s/celeste-%.4f-%.4f-%.4f-%.4f.jld",
                     outdir, ramin, ramax, decmin, decmax)
    JLD.save(fname, "results", results)
end


# -----------------------------------------------------------------------------
# Scoring

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

Load the SDSS photoObj catalog used to initialize celeste, and reformat column
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
function convert(::Type{CatalogEntry}, vs::Vector{Float64}, objid::ASCIIString,
                 thingid::Int)
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
        objid,
        thingid)
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
function celeste_to_df(results::Dict{Int, Dict})
    # Initialize dataframe
    N = length(results)
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
    i = 0
    for (thingid, result) in results
        i += 1
        vs = result["vs"]
        ce = convert(CatalogEntry, vs, result["objid"], thingid)
        load_ce!(i, ce, df)

        df[i, :is_star] = vs[ids.a[1]]

        for j in 1:2
            s_type = ["star", "gal"][j]
            df[i, symbol("$(s_type)_flux_r_sd")] =
                sqrt(df[i, symbol("$(s_type)_flux_r")]) * vs[ids.r2[j]]
            for c in 1:4
                cc_sd = symbol("$(s_type)_color_$(color_names[c])_sd")
                df[i, cc_sd] = 2.5 * log10(e) * vs[ids.c2[c, j]]
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


function score(ra_range::Tuple{Float64, Float64},
               dec_range::Tuple{Float64, Float64},
               fieldid::Tuple{Int, Int, Int},
               results,
               reffile,
               primary_dir)
    # convert Celeste results to a DataFrame.
    celeste_full_df = celeste_to_df(results)
    println("celeste: $(size(celeste_full_df, 1)) objects")

    # load coadd catalog
    coadd_full_df = load_s82(reffile)
    println("coadd catalog: $(size(coadd_full_df, 1)) objects")

    # find matches in coadd catalog by position
    disttol = 1.0 / 3600.0  # 1 arcsec
    good_coadd_indexes = Int[]
    good_celeste_indexes = Int[]
    for i in 1:size(celeste_full_df, 1)
        try
            j = match_position(coadd_full_df[:ra], coadd_full_df[:dec],
                         celeste_full_df[i, :ra], celeste_full_df[i, :dec],
                         disttol)
            push!(good_celeste_indexes, i)
            push!(good_coadd_indexes, j)
        catch e
            println(e)
        end
    end

    celeste_df = celeste_full_df[good_celeste_indexes, :]
    coadd_df = coadd_full_df[good_coadd_indexes, :]

    # load "primary" catalog (the SDSS photoObj catalog used to initialize
    # celeste).
    primary_full_df = load_primary(primary_dir, fieldid...)

    println("primary catalog: $(size(primary_full_df, 1)) objects")

    # match by object id
    good_primary_indexes = Int[findfirst(primary_full_df[:objid], objid)
                   for objid in celeste_df[:objid]]

    # limit primary to matched items
    primary_df = primary_full_df[good_primary_indexes, :]

    # ensure that all objects are matched
    if size(primary_df, 1) != size(celeste_df, 1)
        error("catalog mismatch between celeste and primary")
    end

    # difference between celeste and coadd
    celeste_err = get_err_df(coadd_df, celeste_df)
    primary_err = get_err_df(coadd_df, primary_df)

    println(primary_df[4,:])
    println(primary_err[4,:])
    println("------------------------------")
    println(celeste_df[4,:])
    println(celeste_err[4,:])

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


"""
Score all the celeste results for sources in the given
(`run`, `camcol`, `field`).
This is done by finding all files with names matching
`DIR/celeste-RUN-CAMCOL-FIELD-*.jld`
"""
function score_nersc(ramin, ramax, decmin, decmax, resultdir, reffile)
    # Get vector of (run, camcol, field) triplets overlapping this patch
    fieldids = query_overlapping_fieldids(ramin, ramax, decmin, decmax)

    # find celeste result files matching the pattern
    re = Regex("celeste-.*\.jld")
    fnames = filter(x->ismatch(re, x), readdir(dirname))
    paths = [joinpath(outdir, name) for name in fnames]

    # collect all Celeste results into a single dictionary keyed by
    # thing_id
    results = Dict{Int, Dict}()
    for path in paths
        merge!(results, JLD.load(path, "results"))
    end

    score((ramin, ramax), (decmin, decmax), fieldids,
          results,
          reffile,
          nersc_photoobjdir(fieldid[1], fieldid[2]))
end
