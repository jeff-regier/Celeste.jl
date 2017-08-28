module SampleData

using Celeste: Model, DeterministicVI
import Celeste: Infer
import Celeste: Synthetic
import Celeste.SDSSIO: RunCamcolField, load_field_images, PlainFITSStrategy

using Distributions
using StaticArrays
import WCS, FITSIO, DataFrames

export empty_model_params, datadir,
       sample_ce, perturb_params,
       sample_star_fluxes, sample_galaxy_fluxes,
       gen_sample_star_dataset, gen_sample_galaxy_dataset,
       gen_two_body_dataset, gen_three_body_dataset, gen_n_body_dataset,
       make_elbo_args, true_star_init

const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")

const sample_star_fluxes = [
    4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]
const sample_galaxy_fluxes = [
    1.377666E+01, 5.635334E+01, 1.258656E+02,
    1.884264E+02, 2.351820E+02] * 100  # 1x wasn't bright enough

# A world coordinate system where the world and pixel coordinates are the same.
const wcs_id = WCS.WCSTransform(2,
                    cd = Float64[1 0; 0 1],
                    ctype = ["none", "none"],
                    crpix = Float64[1, 1],
                    crval = Float64[1, 1]);

# Ensure that test images are available.
wd = pwd()
cd(datadir)
run(`make RUN=3900 CAMCOL=6 FIELD=269`)
cd(wd)


const sample_rcf = RunCamcolField(3900, 6, 269)
const sample_images = load_field_images(PlainFITSStrategy(datadir),sample_rcf)


"""
Turn a images and vector of catalog entries into elbo arguments
that can be used with Celeste.
"""
function make_elbo_args(images::Vector{Image},
                        catalog::Vector{CatalogEntry};
                        active_source=-1,
                        patch_radius_pix::Float64=NaN,
                        include_kl=true)
    patches = Infer.get_sky_patches(images,
                                    catalog,
                                    radius_override_pix=patch_radius_pix)
    S = length(catalog)
    active_sources = active_source > 0 ? [active_source] :
                                          S <= 3 ? collect(1:S) : [1,2,3]
    ElboArgs(images, patches, active_sources; include_kl=include_kl)
end


function empty_model_params(S::Int)
    vp = [DeterministicVI.generic_init_source([ 0., 0. ]) for s in 1:S]
    ElboArgs(Image[],
             vp,
             Matrix{SkyPatch}(S, 0),
             collect(1:S))
end


function sample_ce(pos, is_star::Bool)
    CatalogEntry(pos, is_star, sample_star_fluxes, sample_galaxy_fluxes,
        0.1, .7, pi/4, 4., "sample", 0)
end


# for testing away from the truth, where derivatives != 0
function perturb_params(vp)
    for vs in vp
        vs[ids.is_star] = [ 0.4, 0.6 ]
        vs[ids.pos[1]] += .8
        vs[ids.pos[2]] -= .7
        vs[ids.flux_loc] -= log(10)
        vs[ids.flux_scale] *= 25.
        vs[ids.gal_fracdev] += 0.05
        vs[ids.gal_ab] += 0.05
        vs[ids.gal_angle] += pi/10
        vs[ids.gal_scale] *= 1.2
        vs[ids.color_mean] += 0.5
        vs[ids.color_var] =  1e-1
    end
end


function gen_sample_star_dataset(; perturb=true)
    srand(1)
    images0 = deepcopy(sample_images)
    for b in 1:5
        images0[b].H, images0[b].W = 20, 23
        images0[b].wcs = wcs_id
    end
    catalog = [sample_ce([10.1, 12.2], true),]
    images = Synthetic.gen_blob(images0, catalog)
    ea = make_elbo_args(images, catalog)

    vp = Vector{Float64}[DeterministicVI.catalog_init_source(ce) for ce in catalog]
    if perturb
        perturb_params(vp)
    end

    ea, vp, catalog
end


function gen_sample_galaxy_dataset(; perturb=true, include_kl=true)
    srand(1)
    images0 = deepcopy(sample_images)
    for b in 1:5
        images0[b].H, images0[b].W = 20, 23
        images0[b].wcs = wcs_id
    end
    catalog = [sample_ce([8.5, 9.6], false),]
    images = Synthetic.gen_blob(images0, catalog)
    ea = make_elbo_args(images, catalog; include_kl=include_kl)

    vp = Vector{Float64}[DeterministicVI.catalog_init_source(ce) for ce in catalog]
    if perturb
        perturb_params(vp)
    end

    ea, vp, catalog
end

function gen_two_body_dataset(; perturb=true)
    # A small two-body dataset for quick unit testing.  These objects
    # will be too close to be identifiable.

    srand(1)
    images0 = deepcopy(sample_images)
    for b in 1:5
        images0[b].H, images0[b].W = 20, 23
        images0[b].wcs = wcs_id
    end
    catalog = [
        sample_ce([4.5, 3.6], false),
        sample_ce([10.1, 12.1], true)
    ]
    images = Synthetic.gen_blob(images0, catalog)
    ea = make_elbo_args(images, catalog)

    vp = Vector{Float64}[DeterministicVI.catalog_init_source(ce) for ce in catalog]
    if perturb
        perturb_params(vp)
    end

    ea, vp, catalog
end



function gen_three_body_dataset(; perturb=true)
    srand(1)
    images0 = deepcopy(sample_images)
    for b in 1:5
        images0[b].H, images0[b].W = 112, 238
        images0[b].wcs = wcs_id
    end
    catalog = [
        sample_ce([4.5, 3.6], false),
        sample_ce([60.1, 82.2], true),
        sample_ce([71.3, 100.4], false),
    ];
    images = Synthetic.gen_blob(images0, catalog);
    ea = make_elbo_args(images, catalog);

    vp = Vector{Float64}[DeterministicVI.catalog_init_source(ce) for ce in catalog]
    if perturb
        perturb_params(vp)
    end

    ea, vp, catalog
end


"""
Generate a large dataset with S randomly placed bodies and non-constant
background.
"""
function gen_n_body_dataset(
        S::Int; patch_pixel_radius=20., seed=NaN, perturb=true)
    if !isnan(seed)
        srand(seed)
    end

    images0 = deepcopy(sample_images)
    img_size_h = 900
    img_size_w = 1000
    for b in 1:5
        images0[b].H, images0[b].W = img_size_h, img_size_w
    end

    fluxes = [4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]

    locations = rand(2, S) .* [img_size_h, img_size_w]
    world_locations = WCS.pix_to_world(images0[3].wcs, locations)

    catalog = CatalogEntry[CatalogEntry(world_locations[:, s], true,
            fluxes, fluxes, 0.1, .7, pi/4, 4., string(s), s) for s in 1:S];

    images = Synthetic.gen_blob(images0, catalog);

    # Make non-constant background.
    for b=1:5
        images[b].nelec_per_nmgy = fill(images[b].nelec_per_nmgy[1], images[b].H)
        images[b].sky = SkyIntensity(fill(images[b].sky[1,1], images[b].H, images[b].W),
                                     collect(1:images[b].H), collect(1:images[b].W),
                                     ones(images[b].H))
    end

    ea = make_elbo_args(
        images, catalog, patch_radius_pix=patch_pixel_radius)

    vp = Vector{Float64}[DeterministicVI.catalog_init_source(ce) for ce in catalog]
    if perturb
        perturb_params(vp)
    end

    ea, vp, catalog
end


function true_star_init()
    ea, vp, catalog = gen_sample_star_dataset(perturb=false)

    vp[1][ids.is_star] = [ 1.0 - 1e-4, 1e-4 ]
    vp[1][ids.flux_scale] = 1e-4
    vp[1][ids.flux_loc] = log(sample_star_fluxes[3]) - 0.5 * vp[1][ids.flux_scale]
    vp[1][ids.color_var] = 1e-4

    ea, vp, catalog
end


end # End module
