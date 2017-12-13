using Base.Test

using Celeste: Model, Transform, SensitiveFloats, MCMC, Synthetic
using StatsBase

# helper to create synthetic data
function generate_single_star_data(; lnr=7.7251)
    # generate images (data)
    ea, vp, catalog = SampleData.gen_sample_star_dataset()

    # adjust fluxes to match input lnr
    old_fluxes = catalog[1].star_fluxes
    _, colors = Model.fluxes_to_colors(old_fluxes)
    new_fluxes = Model.colors_to_fluxes(lnr, colors)
    catalog[1].star_fluxes = new_fluxes

    # cache fluxes location, etc
    ce0 = deepcopy(catalog[1])
    dat_images = deepcopy(ea.images)

    # make mod/dat images
    Synthetic.gen_images!(dat_images, catalog; expectation=false);

    # make sure ea images data reflects dat images for vb fitting
    for b in 1:5
      ea.images[b].pixels[:] = dat_images[b].pixels[:]
    end

    # save truedf for python plot comparison
    truedf = MCMC.catalog_to_data_frame_row(ce0)
    return truedf, ce0, dat_images, ea, vp
end

###############
#### tests ####
###############

function test_mcmc_catalog_to_data_frame_row()
    # create slightly less bright log re
    truedf, ce0, dat_images, ea, vp = generate_single_star_data(; lnr=6.)
end

function test_star_inference_functions()
    # gen data and initial params
    truedf, ce0, images, ea, vp = generate_single_star_data(; lnr=6.)
    patches = Model.get_sky_patches(images, [ce0]; radius_override_pix=25)
    background_images = [zeros(size(img.pixels)) for img in images]
    star_loglike, star_logprior, star_logpost, sample_star_prior, ra_lim, dec_lim, uniform_to_deg, deg_to_uniform = 
        MCMC.make_star_inference_functions(images, ce0;
                                           patches=patches[1,:], background_images=background_images,
                                           pos_delta = [2., 2.])
    th_rand = sample_star_prior()
    @test !isnan(star_loglike(th_rand))
    @test !isnan(star_logprior(th_rand))
    @test !isnan(star_logpost(th_rand))
end

function test_gal_inference_functions()
    # gen data and initial params
    truedf, ce0, images, ea, vp = generate_single_star_data(; lnr=6.)
    patches = Model.get_sky_patches(images, [ce0]; radius_override_pix=25)
    background_images = [zeros(size(img.pixels)) for img in images]
    gal_loglike, gal_logprior, gal_logpost, sample_gal_prior, ra_lim, dec_lim, uniform_to_deg, deg_to_uniform = 
        MCMC.make_gal_inference_functions(images, ce0;
                                           patches=patches[1,:], background_images=background_images,
                                           pos_delta = [2., 2.])
    th_rand = sample_gal_prior()
    @test !isnan(gal_loglike(th_rand))
    @test !isnan(gal_logprior(th_rand))
    @test !isnan(gal_logpost(th_rand))
end

@testset "MCMC" begin
    test_star_inference_functions()
    test_gal_inference_functions()
end
