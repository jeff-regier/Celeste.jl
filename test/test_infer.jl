## test the main entry point in Celeste: the `infer` function
import JLD

import Celeste: Config
using Celeste.SDSSIO
using Celeste.ParallelRun


@testset "one_node_single_infer_mcmc with a single (run, camcol, field)" begin
    # very small patch of sky that turns out to have 4 sources.
    # We checked that this patch is in the given field.
    box = ParallelRun.BoundingBox(164.39, 164.41, 39.11, 39.13)
    rcfs = [RunCamcolField(3900, 6, 269),]
    strategy = PlainFITSStrategy(datadir)
    ctni = ParallelRun.infer_init(rcfs, strategy; box=box)
    result = ParallelRun.one_node_single_infer(ctni...; config=Config(2.0, 3, 2), do_vi=false)
end


@testset "infer_box runs" begin
    box = ParallelRun.BoundingBox("164.39", "164.41", "39.11", "39.13")
    rcfs = [RunCamcolField(3900, 6, 269),]
    ParallelRun.infer_box(box, datadir, datadir)
end


@testset "one_node_single_infer with a single (run, camcol, field)" begin
    # very small patch of sky that turns out to have 4 sources.
    # We checked that this patch is in the given field.
    box = ParallelRun.BoundingBox(164.39, 164.41, 39.11, 39.13)
    rcfs = [RunCamcolField(3900, 6, 269),]
    strategy = PlainFITSStrategy(datadir)
    ctni = ParallelRun.infer_init(rcfs, strategy; box=box)
    result = ParallelRun.one_node_single_infer(ctni...)
end


@testset "load active pixels" begin
    ea, vp, catalog = gen_sample_star_dataset()

    # these images have 20 * 23 * 5 = 2300 pixels in total.
    # the star is bright but it doesn't cover the whole image.
    # it's hard to say exactly how many pixels should be active,
    # but not all of them, and not none of them.
    config = Config(0.0)
    ParallelRun.load_active_pixels!(config, ea.images, ea.patches)

    # most star light (>90%) should be recorded by the active pixels
    num_active_photons = 0.0
    num_active_pixels = 0
    total_pixels = 0
    for n in 1:ea.N
        img = ea.images[n]
        p = ea.patches[1, n]
        H2, W2 = size(p.active_pixel_bitmap)
        total_pixels += length(img.pixels)
        for w2 in 1:W2, h2 in 1:H2
            # (h2, w2) index the local patch, while (h, w) index the image
            h = p.bitmap_offset[1] + h2
            w = p.bitmap_offset[2] + w2
            num_active_photons += img.pixels[h, w] - img.sky[h, w]
            num_active_pixels += 1
        end
    end

    @test 100 < num_active_pixels < total_pixels

    total_photons = 0.0
    for img in ea.images
        for w in 1:img.W, h in 1:img.H
            total_photons += img.pixels[h, w] - img.sky[h, w]
        end
    end

    @test num_active_photons <= total_photons  # sanity check
    @test num_active_photons > 0.9 * total_photons

    # super dim images
    for img in ea.images
        for w in 1:img.W, h in 1:img.H
            img.pixels[h, w] = img.sky[h, w]
        end
    end

    # only 2 pixels per image are within 0.6 pixels of the
    # source's center (10.9, 11.5)
    config = Config(0.6)
    ParallelRun.load_active_pixels!(config, ea.images, ea.patches)

    for n in 1:ea.N
#  FIXME: is load active pixels off by (0.5, 0.5)?
#        @test sum(ea.patches[1,n].active_pixel_bitmap) == 2
    end
end


@testset "active pixel selection" begin
    ea, vp, catalog = gen_two_body_dataset();
    patches = Model.get_sky_patches(ea.images, catalog; radius_override_pix=5);
    config = Config(5.0)
    ParallelRun.load_active_pixels!(config, ea.images, patches, noise_fraction=Inf)

    for n in 1:ea.N
        # Make sure, for testing purposes, that the whole bitmap isn't full.
        for s in 1:ea.S
            @test sum(patches[s, n].active_pixel_bitmap) <
                  prod(size(patches[s, n].active_pixel_bitmap))
        end

        function patch_in_whole_image(p::SkyPatch)
            patch_image = zeros(size(ea.images[n].pixels))
            for h in 1:ea.images[n].H, w in 1:ea.images[n].W
                if ParallelRun.is_pixel_in_patch(h, w, p)
                    patch_image[h, w] += 1
                end
            end
            return patch_image
        end
        patch_images = [ patch_in_whole_image(patches[s, 3]) for s in 1:ea.S ];
        for s in 1:ea.S
            @test sum(patch_images[s]) == sum(patches[s, n].active_pixel_bitmap)
        end

        H_min, W_min, H_max, W_max =
            ParallelRun.get_active_pixel_range(patches, collect(1:ea.S), n);
        patch_image = zeros(H_max - H_min + 1, W_max - W_min + 1);

        for h in H_min:H_max, w in W_min:W_max, s in 1:ea.S
            p = patches[s, n]
            if ParallelRun.is_pixel_in_patch(h, w, p)
                patch_image[h - H_min + 1, w - W_min + 1] += 1
            end
        end
        @test all(patch_image .== sum(patch_images)[H_min:H_max, W_min:W_max])
    end
end


@testset "don't select a patch that is way too big" begin
    wd = pwd()
    cd(datadir)
    run(`make RUN=4114 CAMCOL=3 FIELD=127`)
    run(`make RUN=4114 CAMCOL=4 FIELD=127`)
    cd(wd)

    rcfs = [RunCamcolField(4114, 3, 127), RunCamcolField(4114, 4, 127)]
    strategy = PlainFITSStrategy(datadir)
    images = SDSSIO.load_field_images(strategy, rcfs)
    catalog = SDSSIO.read_photoobj_files(strategy, rcfs)

    entry_id = 429  # star at RA, Dec = (309.49754066435867, 45.54976572870953)

    neighbors = ParallelRun.find_neighbors([entry_id,], catalog, images)[1]

    # there's a lot near this star, but not a lot that overlaps with it, see
    # http://skyserver.sdss.org/dr10/en/tools/explore/summary.aspx?id=0x112d1012607f050a
    @test length(neighbors) < 5
end
