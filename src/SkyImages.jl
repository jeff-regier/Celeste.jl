module SkyImages

using CelesteTypes
import SloanDigitalSkySurvey: PSF, SDSS, WCSUtils
import SloanDigitalSkySurvey.PSF.get_psf_at_point

import WCS
import DataFrames
import ElboDeriv # For stitch_object_tiles
import FITSIO
import GaussianMixtures
import Grid
import Util

export load_stamp_blob, load_sdss_blob, crop_image!
export convert_gmm_to_celeste, get_psf_at_point
export convert_catalog_to_celeste, load_stamp_catalog
export break_blob_into_tiles, break_image_into_tiles
export get_local_sources
export stitch_object_tiles


@doc """
Load a stamp catalog.
""" ->
function load_stamp_catalog(cat_dir, stamp_id, blob; match_blob=false)
    df = SDSS.load_stamp_catalog_df(cat_dir, stamp_id, blob,
                                    match_blob=match_blob)
    df[:objid] = [ string(s) for s=1:size(df)[1] ]
    convert_catalog_to_celeste(df, blob, match_blob=match_blob)
end


@doc """
Convert a dataframe catalog (e.g. as returned by
SloanDigitalSkySurvey.SDSS.load_catalog_df) to an array of Celeste CatalogEntry
objects.

Args:
  - df: The dataframe output of SloanDigitalSkySurvey.SDSS.load_catalog_df
  - blob: The blob that the catalog corresponds to
  - match_blob: If false, changes the direction of phi to match tractor,
                not Celeste.
""" ->
function convert_catalog_to_celeste(
  df::DataFrames.DataFrame, blob::Array{Image, 1}; match_blob=false)
    function row_to_ce(row)
        x_y = [row[1, :ra], row[1, :dec]]
        star_fluxes = zeros(5)
        gal_fluxes = zeros(5)
        fracs_dev = [row[1, :frac_dev], 1 - row[1, :frac_dev]]
        for b in 1:length(band_letters)
            bl = band_letters[b]
            psf_col = symbol("psfflux_$bl")

            # TODO: How can there be negative fluxes?
            star_fluxes[b] = max(row[1, psf_col], 1e-6)

            dev_col = symbol("devflux_$bl")
            exp_col = symbol("expflux_$bl")
            gal_fluxes[b] += fracs_dev[1] * max(row[1, dev_col], 1e-6) +
                             fracs_dev[2] * max(row[1, exp_col], 1e-6)
        end

        fits_ab = fracs_dev[1] > .5 ? row[1, :ab_dev] : row[1, :ab_exp]
        fits_phi = fracs_dev[1] > .5 ? row[1, :phi_dev] : row[1, :phi_exp]
        fits_theta = fracs_dev[1] > .5 ? row[1, :theta_dev] : row[1, :theta_exp]

        # tractor defines phi as -1 * the phi catalog for some reason.
        if !match_blob
            fits_phi *= -1.
        end

        re_arcsec = max(fits_theta, 1. / 30)  # re = effective radius
        re_pixel = re_arcsec / 0.396

        phi90 = 90 - fits_phi
        phi90 -= floor(phi90 / 180) * 180
        phi90 *= (pi / 180)

        CatalogEntry(x_y, row[1, :is_star], star_fluxes,
            gal_fluxes, row[1, :frac_dev], fits_ab, phi90, re_pixel,
            row[1, :objid])
    end

    CatalogEntry[row_to_ce(df[i, :]) for i in 1:size(df, 1)]
end


@doc """
Load a stamp into a Celeste blob.
""" ->
function load_stamp_blob(stamp_dir, stamp_id)
    function fetch_image(b)
        band_letter = band_letters[b]
        filename = "$stamp_dir/stamp-$band_letter-$stamp_id.fits"

        fits = FITSIO.FITS(filename)
        hdr = FITSIO.read_header(fits[1])
        original_pixels = read(fits[1])
        dn = original_pixels / hdr["CALIB"] + hdr["SKY"]
        nelec_f32 = round(dn * hdr["GAIN"])
        nelec = convert(Array{Float64}, nelec_f32)

        header_str = FITSIO.read_header(fits[1], ASCIIString)
        wcs = WCS.from_header(header_str)[1]
        close(fits)

        alphaBar = [hdr["PSF_P0"]; hdr["PSF_P1"]; hdr["PSF_P2"]]
        xiBar = [
            [hdr["PSF_P3"]  hdr["PSF_P4"]];
            [hdr["PSF_P5"]  hdr["PSF_P6"]];
            [hdr["PSF_P7"]  hdr["PSF_P8"]]]'

        tauBar = Array(Float64, 2, 2, 3)
        tauBar[:,:,1] = [[hdr["PSF_P9"] hdr["PSF_P11"]];
                         [hdr["PSF_P11"] hdr["PSF_P10"]]]
        tauBar[:,:,2] = [[hdr["PSF_P12"] hdr["PSF_P14"]];
                         [hdr["PSF_P14"] hdr["PSF_P13"]]]
        tauBar[:,:,3] = [[hdr["PSF_P15"] hdr["PSF_P17"]];
                         [hdr["PSF_P17"] hdr["PSF_P16"]]]

        psf = [PsfComponent(alphaBar[k], xiBar[:, k],
                            tauBar[:, :, k]) for k in 1:3]

        H, W = size(original_pixels)
        iota = hdr["GAIN"] / hdr["CALIB"]
        epsilon = hdr["SKY"] * hdr["CALIB"]

        run_num = round(Int, hdr["RUN"])
        camcol_num = round(Int, hdr["CAMCOL"])
        field_num = round(Int, hdr["FIELD"])

        Image(H, W, nelec, b, wcs, epsilon, iota, psf,
              run_num, camcol_num, field_num)
    end

    blob = map(fetch_image, 1:5)
end


@doc """
Read a blob from SDSS.

Args:
 - field_dir: The directory of the file
 - run_num: The run number
 - camcol_num: The camcol number
 - field_num: The field number

Returns:
 - A blob (array of Image objects).
""" ->
function load_sdss_blob(field_dir, run_num, camcol_num, field_num;
  mask_planes =
    Set(["S_MASK_INTERP", "S_MASK_SATUR", "S_MASK_CR", "S_MASK_GHOST"]))

    band_gain, band_dark_variance =
      SDSS.load_photo_field(field_dir, run_num, camcol_num, field_num)

    blob = Array(Image, 5)
    for b=1:5
        print("Reading band $b image data...")
        nelec, calib_col, sky_grid, sky_x, sky_y, sky_image, wcs =
            SDSS.load_raw_field(field_dir, run_num, camcol_num,
                                field_num, b, band_gain[b]);

        print("Masking image...")
        SDSS.mask_image!(nelec, field_dir, run_num, camcol_num, field_num, b,
                         mask_planes=mask_planes);
        println("done.")
        H = size(nelec, 1)
        W = size(nelec, 2)

        # For now, use the median noise and sky image.  Here,
        # epsilon * iota needs to be in units comparable to nelec
        # electron counts.
        # Note that each are actuall pretty variable.
        iota = convert(Float64, band_gain[b] / median(calib_col))
        epsilon = convert(Float64, median(sky_image) * median(calib_col))
        epsilon_mat = Array(Float64, H, W)
        iota_vec = Array(Float64, H)
        for h=1:H
            iota_vec[h] = band_gain[b] / calib_col[h]
            for w=1:W
                epsilon_mat[h, w] = sky_image[h, w] * calib_col[h]
            end
        end

        # Load and fit the psf.
        println("reading psf...")
        raw_psf_comp =
          SDSS.load_psf_data(field_dir, run_num, camcol_num, field_num, b);

        # For now, evaluate the psf at the middle of the image.
        psf_point_x = H / 2
        psf_point_y = W / 2

        raw_psf = get_psf_at_point(psf_point_x, psf_point_y, raw_psf_comp);
        psf_gmm, scale = PSF.fit_psf_gaussians(raw_psf);
        psf = convert_gmm_to_celeste(psf_gmm, scale)

        # Set it to use a constant background but include the non-constant data.
        blob[b] = Image(H, W, nelec, b, wcs, epsilon, iota, psf,
                      parse(Int, run_num),
                      parse(Int, camcol_num),
                      parse(Int, field_num),
                      false, epsilon_mat, iota_vec, raw_psf_comp)
    end

    blob
end


@doc """
Crop an image in place to a (2 * width) x (2 * width) - pixel square centered
at the world coordinates wcs_center.
Args:
  - blob: The field to crop
  - width: The width in pixels of each quadrant
  - wcs_center: A location in world coordinates (e.g. the location of a
                celestial body)

Returns:
  - A tiled blob with a single tile in each image centered at wcs_center.
    This can be used to investigate a certain celestial object in a single
    tiled blob, for example.
""" ->
function crop_blob_to_location(
  blob::Array{Image, 1},
  width::Union{Float64, Int64},
  wcs_center::Vector{Float64})
    @assert length(wcs_center) == 2
    @assert width > 0

    tiled_blob = Array(TiledImage, length(blob))
    for b=1:5
        # Get the pixels that are near enough to the wcs_center.
        pix_center = WCSUtils.world_to_pix(blob[b].wcs, wcs_center)
        h_min = max(floor(Int, pix_center[1] - width), 1)
        h_max = min(ceil(Int, pix_center[1] + width), blob[b].H)
        sub_rows_h = h_min:h_max

        w_min = max(floor(Int, (pix_center[2] - width)), 1)
        w_max = min(ceil(Int, pix_center[2] + width), blob[b].W)
        sub_rows_w = w_min:w_max
        tiled_blob[b] = fill(ImageTile(blob[b], sub_rows_h, sub_rows_w), 1, 1)
    end
    tiled_blob
end


############################################
# PSF functions

@doc """
Convert a GaussianMixtures.GMM object to an array of Celect PsfComponents.

Args:
 - gmm: A GaussianMixtures.GMM object (e.g. as returned by fit_psf_gaussians)

 Returns:
  - An array of PsfComponent objects.
""" ->
function convert_gmm_to_celeste(gmm::GaussianMixtures.GMM, scale::Float64)
    function convert_gmm_component_to_celeste(gmm::GaussianMixtures.GMM, d)
        CelesteTypes.PsfComponent(scale * gmm.w[d],
            collect(GaussianMixtures.means(gmm)[d, :]),
            GaussianMixtures.covars(gmm)[d])
    end

    CelesteTypes.PsfComponent[
      convert_gmm_component_to_celeste(gmm, d) for d=1:gmm.n ]
end


@doc """
Return an image of a Celeste GMM PSF evaluated at rows, cols.

Args:
 - psf_array: The PSF to be evaluated as an array of PsfComponent
 - rows: The rows in the image (usually in pixel coordinates)
 - cols: The column in the image (usually in pixel coordinates)

 Returns:
  - The PSF values at rows and cols.  The default size is the same as
    that returned by get_psf_at_point applied to FITS header values.

Note that the point in the image at which the PSF is evaluated --
that is, the center of the image returned by this function -- is
already implicit in the value of psf_array.
""" ->
function get_psf_at_point(psf_array::Array{CelesteTypes.PsfComponent, 1};
                          rows=collect(-25:25), cols=collect(-25:25))

    function get_psf_value(
      psf::CelesteTypes.PsfComponent, row::Float64, col::Float64)
        x = Float64[row, col] - psf.xiBar
        exp_term = exp(-0.5 * x' * psf.tauBarInv * x - 0.5 * psf.tauBarLd)
        (psf.alphaBar * exp_term / (2 * pi))[1]
    end

    Float64[
      sum([ get_psf_value(psf, float(row), float(col)) for psf in psf_array ])
        for row in rows, col in cols ]
end


@doc """
Get the PSF located at a particular world location in an image.

Args:
  - world_loc: A location in world coordinates.
  - img: An Image

Returns:
  - An array of PsfComponent objects that represents the PSF as a mixture
    of Gaussians.
""" ->
function get_source_psf(world_loc::Vector{Float64}, img::Image)
    # Some stamps or simulated data have no raw psf information.  In that case,
    # just use the psf from the image.
    if size(img.raw_psf_comp.rrows) == (0, 0)
      return img.psf
    else
      pixel_loc = WCSUtils.world_to_pix(img.wcs, world_loc)
      raw_psf =
        PSF.get_psf_at_point(pixel_loc[1], pixel_loc[2], img.raw_psf_comp);
      fit_psf, scale = PSF.fit_psf_gaussians(raw_psf)
      return SkyImages.convert_gmm_to_celeste(fit_psf, scale)
    end
end


#######################################
# Tiling functions

@doc """
Convert an image to an array of tiles of a given width.

Args:
  - img: An image to be broken into tiles
  - tile_width: The size in pixels of each tile

Returns:
  An array of tiles containing the image.
""" ->
function break_image_into_tiles(img::Image, tile_width::Int64)
  WW = ceil(Int, img.W / tile_width)
  HH = ceil(Int, img.H / tile_width)
  ImageTile[ ImageTile(hh, ww, img, tile_width) for hh=1:HH, ww=1:WW ]
end


@doc """
Break a blob into tiles.
""" ->
function break_blob_into_tiles(blob::Blob, tile_width::Int64)
  [ break_image_into_tiles(img, tile_width) for img in blob ]
end


#######################################
# Functions for matching sources to tiles.

# A pixel circle maps locally to a world ellipse.  Return the major
# axis of that ellipse.
function pixel_radius_to_world(pix_radius::Float64,
                               wcs_jacobian::Matrix{Float64})
  pix_radius / minimum(abs(eig(wcs_jacobian)[1]));
end


# A world circle maps locally to a pixel ellipse.  Return the major
# axis of that ellipse.
function world_radius_to_pixel(world_radius::Float64,
                               wcs_jacobian::Matrix{Float64})
  world_radius * maximum(abs(eig(wcs_jacobian)[1]));
end


import SloanDigitalSkySurvey.WCSUtils.world_to_pix
world_to_pix{T <: Number}(patch::SkyPatch, world_loc::Vector{T}) =
    world_to_pix(patch.wcs_jacobian, patch.center, patch.pixel_center,
                 world_loc)


@doc """
A fast function to determine which sources might belong to which tiles.

Args:
  - tiles: A TiledImage
  - patches: A vector of patches (e.g. for a particular band)

Returns:
  - An array (over tiles) of a vector of candidate
    source patches.  If a patch is a candidate, it may be within the patch radius
    of a point in the tile, though it might not.
""" ->
function local_source_candidates(
    tiles::TiledImage, patches::Vector{SkyPatch})

  # Get the largest size of the pixel ellipse defined by the patch
  # world coordinate circle.
  patch_pixel_radii =
    Float64[patches[s].radius * maximum(abs(eig(patches[s].wcs_jacobian)[1]))
            for s=1:length(patches)];

  candidates = fill(Int64[], size(tiles));
  for h=1:size(tiles)[1], w=1:size(tiles)[2]
    # Find the patches that are less than the radius plus diagonal from the
    # center of the tile.  These are candidates for having some
    # overlap with the tile.
    tile = tiles[h, w]
    tile_center = [ mean(tile.h_range), mean(tile.w_range)]
    tile_diag = (0.5 ^ 2) * (tile.h_width ^ 2 + tile.w_width ^ 2)
    patch_distances =
      [ sum((tile_center .- patches[s].pixel_center) .^ 2) for
        s=1:length(patches)]
    candidates[h, w] =
      find(patch_distances .<= (tile_diag .+ patch_pixel_radii) .^ 2)
  end

  candidates
end


@doc """
Args:
  - tile: An ImageTile (containing tile coordinates)
  - patches: A vector of SkyPatch objects to be matched with the tile.

Returns:
  - A vector of source ids (from 1 to length(patches)) that influence
    pixels in the tile.  A patch influences a tile if
    there is any overlap in their squares of influence.
""" ->
function get_local_sources(tile::ImageTile, patches::Vector{SkyPatch})

    tile_sources = Int64[]
    tile_center = Float64[mean(tile.h_range), mean(tile.w_range)]

    for patch_index in 1:length(patches)
      patch = patches[patch_index]
      patch_radius_px = world_radius_to_pixel(patch.radius, patch.wcs_jacobian)

      # This is a "ball" in the infinity norm.
      if (abs(tile_center[1] - patch.pixel_center[1]) <
          patch_radius_px + 0.5 * tile.h_width) &&
         (abs(tile_center[2] - patch.pixel_center[2]) <
          patch_radius_px + 0.5 * tile.w_width)
        push!(tile_sources, patch_index)
      end
    end

    tile_sources
end


@doc """
Return a vector of (h, w) indices of tiles that contain this source.
""" ->
function find_source_tiles(s::Int64, b::Int64, mp::ModelParams)
  [ ind2sub(size(mp.tile_sources[b]), ind) for ind in
    find([ s in sources for sources in mp.tile_sources[b]]) ]
end

@doc """
Combine the tiles associated with a single source into an image.

Args:
  s: The source index
  b: The band
  mp: The ModelParams object
  tiled_blob: The original tiled blob
  predicted: If true, render the object based on the values in ModelParams.
             Otherwise, show the image from tiled_blob.

Returns:
  A matrix of pixel values for the particular object using only tiles in
  which it is found according to the ModelParams tile_sources field.  Pixels
  from tiles that do not have this source will be marked as 0.0.
""" ->
function stitch_object_tiles(
    s::Int64, b::Int64, mp::ModelParams{Float64}, tiled_blob::TiledBlob;
    predicted::Bool=false)

  H, W = size(tiled_blob[b])
  has_s = Bool[ s in mp.tile_sources[b][h, w] for h=1:H, w=1:W ];
  tiles_s = tiled_blob[b][has_s];
  tile_sources_s = mp.tile_sources[b][has_s];
  h_range = Int[typemax(Int), typemin(Int)]
  w_range = Int[typemax(Int), typemin(Int)]
  print("Stitching...")
  for tile in tiles_s
    print(".")
    h_range[1] = min(minimum(tile.h_range), h_range[1])
    h_range[2] = max(maximum(tile.h_range), h_range[2])
    w_range[1] = min(minimum(tile.w_range), w_range[1])
    w_range[2] = max(maximum(tile.w_range), w_range[2])
  end
  println("Done.")

  image_s = fill(0.0, diff(h_range)[1] + 1, diff(w_range)[1] + 1);
  for tile_ind in 1:length(tiles_s)
    tile = tiles_s[tile_ind]
    tile_sources = tile_sources_s[tile_ind]
    image_s[tile.h_range - h_range[1] + 1, tile.w_range - w_range[1] + 1] =
      predicted ?
      ElboDeriv.tile_predicted_image(tile, mp, include_epsilon=false):
      tile.pixels
  end
  image_s
end


end
