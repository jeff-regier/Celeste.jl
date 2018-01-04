# detection (aka segmentation) - detecting sources on images

import WCS
import .SEP
using .Coordinates: match_coordinates
using .Model

# return the world coordinates of all objects in the catalog as a 2xN matrix
# where `result[:, i]` is the (latitude, longitude) of the i-th object.
function _worldcoords(catalog::SEP.Catalog, wcs::WCS.WCSTransform)
    pixcoords = Array{Float64}(2, length(catalog.x))
    for i in eachindex(catalog.x)
        pixcoords[1, i] = catalog.x[i]
        pixcoords[2, i] = catalog.y[i]
    end
    return WCS.pix_to_world(wcs, pixcoords)
end


# Get angle offset between +x axis and +Dec axis (North) from a WCS transform.
# This assumes there is no skew, meaning the x and y axes are perpindicular in
# world coordinates. It is also only based on the CD matrix.
function _x_vs_n_angle(wcs::WCS.WCSTransform)
    cd = wcs[:cd]::Matrix{Float64}
    sgn = sign(det(cd))
    n_vs_y_rot = atan2(sgn * cd[1, 2],  sgn * cd[1, 1])  # angle of N CCW
                                                         # from +y axis
    return -(n_vs_y_rot + pi/2)  # angle of +x CCW from N
end


"""
    detect_sources(images)

Detect sources in a set of (possibly overlapping) `Image`s and combine
duplicates. Returns a catalog and a 2-d array of `ImagePatch`.

"""
function detect_sources(images::Vector{<:Image})

    catalogs = SEP.Catalog[]
    for image in images

        calpixels = Model.calibrated_pixels(image)

        # Run background, just to get background rms.
        #
        # We're using sky subtracted (and calibrated) image data, but,
        # we still run a background analysis, just to determine the
        # rough image noise for the purposes of setting a
        # threshold. This could be slightly suboptimal in terms of
        # selecting faint sources: If the image noise varies
        # significantly across the image, the threshold might be too
        # high or too low in some places. However, we don't really
        # trust the variable background RMS without first masking
        # sources. (We could add this.)
        bkg = SEP.Background(calpixels; boxsize=(256, 256),
                             filtersize=(3, 3))
        noise = SEP.global_rms(bkg)
        push!(catalogs, SEP.extract(calpixels, 1.3; noise=noise))
    end

    # The image catalogs only have pixel positions. To match objects in
    # different images, we need to convert these pixel positions to world
    # coordinates in each image.
    catalogs_worldcoords = [_worldcoords(catalog, image.wcs)
                            for (catalog, image) in zip(catalogs, images)]

    # Initialize the "joined" catalog to all the coordinates in the first
    # image. `detections` tracks image index and object index of detections.
    joined_ra = length(images) > 0 ? catalogs_worldcoords[1][1, :] : Float64[]
    joined_dec = length(images) > 0 ? catalogs_worldcoords[1][2, :] : Float64[]
    detections = (length(images) > 0 ?
                  [[(1, j)] for j in 1:length(catalogs[1].x)] :
                  Vector{Tuple{Int, Int}}[])

    # Search for matches in remaining images
    for i in 2:length(images)
        ra = catalogs_worldcoords[i][1, :]
        dec = catalogs_worldcoords[i][2, :]
        idx, dist = match_coordinates(ra, dec, joined_ra, joined_dec)
        for j in eachindex(idx)
            # If there is an object in the joined catalog within 1 arcsec,
            # add the (image, object) index to detections for that object.
            # Otherwise, add the position and (image, object) index as a new
            # object in the joined catalog.
            if dist[j] < (1.0 / 3600.0)
                push!(detections[idx[j]], (i, j))
            else
                push!(joined_ra, ra[j])
                push!(joined_dec, dec[j])
                push!(detections, [(i, j)])
            end
        end
    end

    # Initialize joined output catalog and patches
    nobjects = length(joined_ra)
    result = Array{CatalogEntry}(nobjects)
    patches = Array{ImagePatch}(nobjects, length(images))

    # Precalculate some angles
    x_vs_n_angles = [_x_vs_n_angle(image.wcs) for image in images]

    # Loop over output catalog:
    # - Create catalog entry based on "best" detection.
    # - Create patches based on detection in each image.
    for i in 1:nobjects
        world_center = [joined_ra[i], joined_dec[i]]

        # which detection in each band is the "best" (has most pixels)
        # We use this to initialize flux in each band
        best = fill((0, 0), NUM_BANDS)
        npix = fill(0, NUM_BANDS)
        for (j, catidx) in detections[i]
            b = images[j].b
            np = catalogs[j].npix[catidx]
            if np > npix[b]
                best[b] = (j, catidx)
                npix[b] = np
            end
        end

        # set gal_fluxes (and star_fluxes) to best detection flux or 0 if not
        # detected.
        gal_fluxes = [j != 0 ? catalogs[j].flux[catidx] : 0.0
                      for (j, catidx) in best]
        star_fluxes = copy(gal_fluxes)

        # use the single best band for shape parameters
        j, catidx = best[indmax(npix)]
        gal_axis_ratio = catalogs[j].b[catidx] / catalogs[j].a[catidx]

        # SEP angle is CCW from +x axis and in [-pi/2, pi/2].
        # Add offset of x axis from N to make angle CCW from N
        gal_angle = catalogs[j].theta[catidx] + x_vs_n_angles[j]

        # A 2-d symmetric gaussian has CDF(r) =  1 - exp(-(r/sigma)^2/2)
        # The half-light radius is then r = sigma * sqrt(2 ln(2))
        sigma = sqrt(catalogs[j].a[catidx] * catalogs[j].b[catidx])
        gal_radius_px = sigma * sqrt(2.0 * log(2.0))

        result[i] = CatalogEntry(world_center,
                                 false,  # is_star
                                 star_fluxes,
                                 gal_fluxes,
                                 0.5,  # gal_frac_dev
                                 gal_axis_ratio,
                                 gal_angle,
                                 gal_radius_px)

        # create patches based on detections
        for (j, catidx) in detections[i]
            box = (Int(catalogs[j].xmin[catidx]):Int(catalogs[j].xmax[catidx]),
                   Int(catalogs[j].ymin[catidx]):Int(catalogs[j].ymax[catidx]))
            box = Model.dilate_box(box, 0.2)
            minbox = Model.box_around_point(images[j].wcs, world_center, 5.0)
            patches[i, j] = ImagePatch(images[j], Model.enclose_boxes(box, minbox))
        end

        # fill patches for images where there was no detection
        for j in 1:length(images)
            if !isassigned(patches, i, j)
                box = Model.box_around_point(images[j].wcs, world_center, 5.0)
                patches[i, j] = ImagePatch(images[j], box)
            end
        end
    end

    return result, patches
end
