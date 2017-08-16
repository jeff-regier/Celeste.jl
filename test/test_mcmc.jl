using Base.Test

using Celeste: Model, Transform, SensitiveFloats, MCMC, Synthetic
using StatsBase

include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))


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
    ce0    = deepcopy(catalog[1])
    images = deepcopy(ea.images)

    # make mod/dat images
    dat_images = Synthetic.gen_blob(images, catalog; expectation=false);

    # make sure ea images data reflects dat images for vb fitting
    for b in 1:5
      ea.images[b].pixels[:] = dat_images[b].pixels[:]
    end

    # save truedf for python plot comparison
    truedf = MCMC.catalog_to_data_frame_row(ce0)
    return truedf, ce0, dat_images, ea, vp
end

function generate_params()
    lnfluxes = MCMC.sample_logfluxes(; is_star=true, lnr=nothing)
    lu       = .01 * randn(2)
    return vcat([lnfluxes, lu]...)
end


###############
#### tests ####
###############


function test_mcmc_catalog_to_data_frame_row()
    # create slightly less bright log re
    truedf, ce0, dat_images, ea, vp = generate_single_star_data(; lnr=6.)
    @test true
end


function test_star_loglike()
    # gen data and initial params
    truedf, ce0, dat_images, ea, vp = generate_single_star_data(; lnr=6.)

    # generate star params
    lnfluxes = MCMC.sample_logfluxes(; is_star=true, lnr=nothing)
    lu       = .01 * randn(2)
    th = vcat([lnfluxes, lu]...)

    # create loglike
    init_pos = deepcopy(vp[1][ids.u])
    star_loglike, constrain_pos, unconstrain_pos =
        MCMC.make_star_loglike(dat_images, init_pos)

    ll = star_loglike(th)
    @test !isnan(ll)

    # make sure constriain/unconstrain work
    uu = constrain_pos(lu)
    lu2 = unconstrain_pos(uu)
    @test isapprox(lu, lu2)

end


function test_logflux_logprior()
    # test sample (with all args)
    lnfluxes = MCMC.sample_logfluxes(; is_star=true, lnr=nothing)
    lnfluxes = MCMC.sample_logfluxes(; is_star=false, lnr=nothing)
    lnfluxes = MCMC.sample_logfluxes(; is_star=true, lnr=5.)
    lnfluxes = MCMC.sample_logfluxes(; is_star=false, lnr=5.)

    ll = MCMC.logflux_logprior(lnfluxes; is_star=true)
    @test !isnan(ll)

    ll = MCMC.logflux_logprior(lnfluxes; is_star=false)
    @test !isnan(ll)
end


function test_slicesample()
    function lnpdf(th)
        return -1*sum(th.*th)
    end

    th = randn(5)
    chain, lls = MCMC.slicesample_chain(lnpdf, th, 10; print_skip=20)
    @test true
end


println("Running mcmc tests")
test_mcmc_catalog_to_data_frame_row()
test_star_loglike()
test_logflux_logprior()
test_slicesample()
