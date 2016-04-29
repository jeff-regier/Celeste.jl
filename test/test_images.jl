using Base.Test
using DataFrames
import WCS

println("Running SkyImages tests.")

const RUN = 3900
const CAMCOL = 6
const FIELD = 269



"""
Make a copy of a ElboArgs keeping only some sources.
"""
function copy_mp_subset{T <: Number}(ea_all::ElboArgs{T}, keep_s::Vector{Int})
    ea = ElboArgs(deepcopy(ea_all.vp[keep_s]))
    ea.active_sources = Int[]
    ea.patches = Array(SkyPatch, ea.S, size(ea_all.patches, 2))

    # Indices of sources in the new model params
    for sa in 1:length(keep_s)
        s = keep_s[sa]
        ea.patches[sa, :] = ea_all.patches[s, :]
        if s in ea_all.active_sources
            push!(ea.active_sources, sa)
        end
    end

    @assert length(ea_all.tile_sources) == size(ea_all.patches, 2)
    num_bands = length(ea_all.tile_sources)
    ea.tile_sources = Array(Matrix{Vector{Int}}, num_bands)
    for b=1:num_bands
        ea.tile_sources[b] = Array(Vector{Int}, size(ea_all.tile_sources[b]))
        for tile_ind in 1:length(ea_all.tile_sources[b])
                tile_s = intersect(ea_all.tile_sources[b][tile_ind], keep_s)
                ea.tile_sources[b][tile_ind] =
                    Int[ findfirst(keep_s, s) for s in tile_s ]
        end
    end

    ea
end


"""
Get the PSF located at a particular world location in an image.

Args:
 - world_loc: A location in world coordinates.
 - img: An TiledImage

Returns:
 - An array of PsfComponent objects that represents the PSF as a mixture
     of Gaussians.
"""
function get_source_psf(world_loc::Vector{Float64}, img::TiledImage)
    # Some stamps or simulated data have no raw psf information.    In that case,
    # just use the psf from the image.
    if size(img.raw_psf_comp.rrows) == (0, 0)
        # Also return a vector of empty psf params
        return img.psf, fill(fill(NaN, length(PsfParams)), psf_K)
    else
        pixel_loc = WCS.world_to_pix(img.wcs, world_loc)
        psfstamp = img.raw_psf_comp(pixel_loc[1], pixel_loc[2])
        return PSF.fit_raw_psf_for_celeste(psfstamp)
    end
end



"""
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
"""
function crop_blob_to_location(
    blob::Array{Image, 1},
    width::Int,
    wcs_center::Vector{Float64})
        @assert length(wcs_center) == 2
        @assert width > 0

        tiled_blob = Array(TiledImage, length(blob))
        for b=1:length(blob)
                # Get the pixels that are near enough to the wcs_center.
                pix_center = WCS.world_to_pix(blob[b].wcs, wcs_center)
                h_min = max(floor(Int, pix_center[1] - width), 1)
                h_max = min(ceil(Int, pix_center[1] + width), blob[b].H)
                sub_rows_h = h_min:h_max

                w_min = max(floor(Int, (pix_center[2] - width)), 1)
                w_max = min(ceil(Int, pix_center[2] + width), blob[b].W)
                sub_rows_w = w_min:w_max
                tiled_blob[b] = TiledImage(blob[b], tile_width=width)
                tiled_blob[b].tiles = fill(ImageTile(blob[b], sub_rows_h, sub_rows_w), 1, 1)
        end
        tiled_blob
end


function test_blob()
    # A lot of tests are in a single function to avoid having to reload
    # the full image multiple times.
    blob = SDSSIO.load_field_images(RUN, CAMCOL, FIELD, datadir)

    fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" datadir RUN CAMCOL FIELD
    cat_entries = SDSSIO.read_photoobj_celeste(fname)

    tiled_blob, ea = initialize_celeste(blob, cat_entries,
                                                                                                patch_radius=1e-6,
                                                                                                fit_psf=false, tile_width=20)

    # Just check some basic facts about the catalog.
    @test length(cat_entries) == 805
    @test 0 < sum([ce.is_star for ce in cat_entries]) < 805

    # Find an object near the middle of the image.
    ctr = WCS.pix_to_world(blob[3].wcs, [blob[3].H / 2, blob[3].W / 2])
    dist = [sqrt((ce.pos[1] - ctr[1])^2 + (ce.pos[2] - ctr[2])^2)
                    for ce in cat_entries]
    obj_index = findmin(dist)[2]    # index of closest object
    obj_loc = cat_entries[obj_index].pos    # location of closest object

    # Test cropping.
    width = 5
    cropped_blob = crop_blob_to_location(blob, width, obj_loc);
    for b=1:length(blob)
        # Check that it only has one tile of the right size containing the object.
        @assert length(cropped_blob[b].tiles) == 1
        patches = vec(ea.patches[:, b])
        tile_sources = Model.get_local_sources(cropped_blob[b].tiles[1],
                                               Model.patch_ctrs_pix(patches),
                                               Model.patch_radii_pix(patches))
        @test obj_index in tile_sources
    end

    # Test get_source_psf at point while we have the blob loaded.
    test_b = 3
    img = tiled_blob[test_b]
    ea_obj = ModelInit.initialize_model_params(tiled_blob,
                                               cat_entries[obj_index:obj_index])
    pixel_loc = WCS.world_to_pix(img.wcs, obj_loc);
    original_psf_val = img.raw_psf_comp(pixel_loc[1], pixel_loc[2])

    original_psf_celeste = PSF.fit_raw_psf_for_celeste(original_psf_val)[1];
    fit_original_psf_val = PSF.get_psf_at_point(original_psf_celeste);

    obj_psf = get_source_psf(ea_obj.vp[1][ids.u], img)[1];
    obj_psf_val = PSF.get_psf_at_point(obj_psf);

    # The fits should match exactly.
    @test_approx_eq_eps(obj_psf_val, fit_original_psf_val, 1e-6)

    # The raw psf will not be as good.
    @test_approx_eq_eps(obj_psf_val, original_psf_val, 1e-2)

    ea_several =
        ModelInit.initialize_model_params(
            tiled_blob, [cat_entries[1]; cat_entries[obj_index]]);

    # The second set of vp is the object of interest
    point_patch_psf = PSF.get_psf_at_point(ea_several.patches[2, test_b].psf);
    @test_approx_eq_eps(obj_psf_val, point_patch_psf, 1e-6)
end


function test_stamp_get_object_psf()
    stamp_blob, stamp_mp, body = gen_sample_star_dataset();
    img = TiledImage(stamp_blob[3]);
    obj_index =    stamp_mp.vp[1][ids.u]
    pixel_loc = WCS.world_to_pix(img.wcs, obj_index)
    original_psf_val = PSF.get_psf_at_point(img.psf);

    obj_psf_val =
        PSF.get_psf_at_point(get_source_psf(stamp_mp.vp[1][ids.u], img)[1])
    @test_approx_eq_eps(obj_psf_val, original_psf_val, 1e-6)
end


function test_get_tiled_image_source()
    # Test that an object only occurs the appropriate tile's local sources.
    blob, ea, body, tiled_blob = gen_sample_star_dataset();

    ea = ModelInit.initialize_model_params(
        tiled_blob, body; patch_radius=1e-6)

    tiled_img = TiledImage(blob[3], tile_width=10);
    for hh in 1:size(tiled_img.tiles, 1), ww in 1:size(tiled_img.tiles, 2)
        tile = tiled_img.tiles[hh, ww]
        loc = Float64[mean(tile.h_range), mean(tile.w_range)]
        for b = 1:5
            ea.vp[1][ids.u] = loc
            pixel_center = WCSUtils.world_to_pix(blob[b].wcs, loc)
            wcs_jacobian = WCSUtils.pixel_world_jacobian(blob[b].wcs, pixel_center)
            radius_pix = maxabs(eigvals(wcs_jacobian)) * 1e-6
            ea.patches[1, b] = SkyPatch(loc,
                                                                    radius_pix,
                                                                    blob[b].psf,
                                                                    wcs_jacobian,
                                                                    pixel_center)
        end
        patches = vec(ea.patches[:, 3])
        local_sources = Model.get_sources_per_tile(tiled_img.tiles,
                                                                                             Model.patch_ctrs_pix(patches),
                                                                                             Model.patch_radii_pix(patches))
        @test local_sources[hh, ww] == Int[1]
        for hh2 in 1:size(tiled_img.tiles, 1), ww2 in 1:size(tiled_img.tiles, 2)
            if (hh2 != hh) || (ww2 != ww)
                @test local_sources[hh2, ww2] == Int[]
            end
        end
    end
end


function test_local_source_candidate()
    blob, ea, body, tiled_blob = gen_n_body_dataset(100);

    # This is run by gen_n_body_dataset but put it here for safe testing in
    # case that changes.
    ea = ModelInit.initialize_model_params(tiled_blob, body);

    for b=1:length(tiled_blob)
        # Get the sources by iterating over everything.
        patches = vec(ea.patches[:,b])

        tile_sources = Model.get_sources_per_tile(tiled_blob[b].tiles,
                                                                                            Model.patch_ctrs_pix(patches),
                                                                                            Model.patch_radii_pix(patches))

        # Check that all the actual sources are candidates and that this is the
        # same as what is returned by initialize_model_params.
        HH, WW = size(tile_sources)
        for h=1:HH, w=1:WW
            # Get a set of candidates.
            candidates = Model.local_source_candidates(
                                                tiled_blob[b].tiles[h, w],
                                                Model.patch_ctrs_pix(patches),
                                                Model.patch_radii_pix(patches))
            @test setdiff(tile_sources[h, w], candidates) == []
            @test tile_sources[h, w] == ea.tile_sources[b][h, w]
        end
    end
end


function test_set_patch_size()
    # Test that the patch size gets most of the light from a variety of
    # galaxy shapes.
    # This shows that the current patch size is actually far too conservative.

    function gal_catalog_from_scale(gal_scale::Float64, flux_scale::Float64)
        CatalogEntry[CatalogEntry(world_location, false,
                                   flux_scale * fluxes, flux_scale * fluxes,
                                   0.1, .01, pi/4, gal_scale, "sample", 0) ]
    end

    srand(1)
    blob0 = SampleData.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf");
    img_size = 150
    for b in 1:5
            blob0[b].H, blob0[b].W = img_size, img_size
    end
    fluxes = [4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]

    world_location = WCS.pix_to_world(blob0[3].wcs,
                                      Float64[img_size / 2, img_size / 2])

    for gal_scale in [1.0, 10.0], flux_scale in [0.1, 10.0]
        cat = gal_catalog_from_scale(gal_scale, flux_scale);
        blob = Synthetic.gen_blob(blob0, cat);
        tiled_blob, ea =
            initialize_celeste(blob, cat, tile_width=typemax(Int));

        for b=1:length(blob)
            @assert size(tiled_blob[b].tiles) == (1, 1)
            tile_image = ElboDeriv.tile_predicted_image(
                tiled_blob[b].tiles[1,1], ea, ea.tile_sources[b][1,1]);

            pixel_center = WCS.world_to_pix(blob[b].wcs, cat[1].pos)
            radius = Model.choose_patch_radius(
                pixel_center, cat[1], blob[b].psf, tiled_blob[b])

            circle_pts = fill(false, blob[b].H, blob[b].W);
            in_circle = 0.0
            for x=1:size(tile_image)[1], y=1:size(tile_image)[2]
                if ((x - pixel_center[1]) ^ 2 + (y - pixel_center[2]) ^ 2) < radius ^ 2
                    in_circle += tile_image[x, y]
                    circle_pts[x, y] = true
                end
            end
            @test in_circle / sum(tile_image) > 0.95

            # Convenient for visualizing:
            # using PyPlot
            # in_circle / sum(tile_image)
            # imshow(tile_image); colorbar()
            # imshow(circle_pts, alpha=0.4)
        end
    end
end


function test_copy_model_params()
    # A lot of tests are in a single function to avoid having to reload
    # the full image multiple times.
    images = SDSSIO.load_field_images(RUN, CAMCOL, FIELD, datadir);

    # Make sure that ElboArgs can handle more than five images (issue #203)
    push!(images, deepcopy(images[1]));
    fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" datadir RUN CAMCOL FIELD
    cat_entries = SDSSIO.read_photoobj_celeste(fname);

    tiled_images, ea_all =
        initialize_celeste(images, cat_entries, fit_psf=false, tile_width=20);

    # Pick a single object of interest.
    obj_s = 100
    neighbor_map = Infer.find_neighbors([obj_s], cat_entries, tiled_images)
    relevant_sources = vcat(obj_s, neighbor_map[1])

    ea_all.active_sources = [ obj_s ];
    ea = copy_mp_subset(ea_all, relevant_sources);

    # Fit with both and make sure you get the same answer.
    Infer.fit_object_psfs!(ea_all, relevant_sources, tiled_images);
    Infer.fit_object_psfs!(ea, collect(1:ea.S), tiled_images);

    @test ea.S == length(relevant_sources)
    for sa in 1:length(relevant_sources)
        s = relevant_sources[sa]
        @test_approx_eq ea.vp[sa] ea_all.vp[s]
    end

    elbo_all = ElboDeriv.elbo(tiled_images, ea_all);
    elbo = ElboDeriv.elbo(tiled_images, ea);

    @test_approx_eq elbo_all.v elbo.v
    @test_approx_eq elbo_all.d elbo.d
    @test_approx_eq elbo_all.h elbo.h
end


test_copy_model_params()
test_blob()
test_stamp_get_object_psf()
test_get_tiled_image_source()
test_local_source_candidate()
test_set_patch_size()
