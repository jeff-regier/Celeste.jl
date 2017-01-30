using Celeste:
    Model, DeterministicVI, CelesteEDA, DeterministicVIImagePSF, SDSSIO,
    Infer, PSF

# NB: PyPlot is not in REQUIRE.
using PyPlot

##########
# Load data.

const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
rcf = SDSSIO.RunCamcolField(4263, 5, 119)
images = SDSSIO.load_field_images(rcf, datadir);
catalog = SDSSIO.read_photoobj_files([rcf], datadir, duplicate_policy=:first);

# Pick an object.
for cat in catalog
    if minimum(cat.star_fluxes) > 300
        print(cat.objid)
        print("\n")
    end
end
objid = "1237663784734491145"
objids = [ce.objid for ce in catalog];
sa = findfirst(objids, objid);
neighbors = Infer.find_neighbors([sa], catalog, images)[1];
cat_local = vcat(catalog[sa], catalog[neighbors]);
vp = DeterministicVI.init_sources([1], cat_local)
patches = Infer.get_sky_patches(images, cat_local);


###########################
# Visualize a source at initial parameters using both classic and FFT Celeste.

ea = ElboArgs(images, deepcopy(vp), patches, [1]);
Celeste.Infer.load_active_pixels!(ea.images, ea.patches);

ea_fft = ElboArgs(images, deepcopy(vp), patches, [1], psf_K=1);
Celeste.Infer.load_active_pixels!(ea_fft.images, ea.patches; exclude_nan=false);

psf_image_mat = Matrix{Float64}[
    PSF.get_psf_at_point(ea.patches[s, b].psf) for s in 1:ea_fft.S, b in 1:ea_fft.N];
fsm_mat = DeterministicVIImagePSF.FSMSensitiveFloatMatrices[
    DeterministicVIImagePSF.FSMSensitiveFloatMatrices() for
    s in 1:ea_fft.S, b in 1:ea_fft.N];
DeterministicVIImagePSF.initialize_fsm_sf_matrices!(fsm_mat, ea_fft, psf_image_mat);

s = 1
n = 3

image_fft = CelesteEDA.render_source_fft(ea, fsm_mat, s, n, include_iota=false, field=:E_G);
image_orig = CelesteEDA.render_source(ea, s, n, include_iota=false, field=:E_G);

PyPlot.close("all")
matshow(image_fft); colorbar(); title("fft Celeste")
matshow(image_orig); colorbar(); title("original Celeste")


##############################################
# Optimize the same source with both methods.

elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_mat);
f_evals_fft, max_f_fft, max_x_fft, nm_result_fft, transform_fft =
    DeterministicVI.maximize_f(elbo_fft_opt, ea_fft, verbose=true, max_iters=100);
vp_opt_fft = deepcopy(ea_fft.vp[1]);

f_evals, max_f, max_x, nm_result, transform =
    DeterministicVI.maximize_f(DeterministicVI.elbo, ea, verbose=true);
vp_opt = deepcopy(ea.vp[1]);

max_f_fft
max_f

println("FFT ELBO improvement")
(max_f_fft - max_f)  / abs(max_f)

println("FFT Extra iterations")
(nm_result_fft.iterations - nm_result.iterations) / nm_result.iterations