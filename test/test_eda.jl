import SampleData

import ..DeterministicVIImagePSF:
    FSMSensitiveFloatMatrices, initialize_fsm_sf_matrices!, ElboArgs
import ..Infer: get_sky_patches, load_active_pixels!
import ..PSF: get_psf_at_point

function test_render()
    images, ea, two_body = SampleData.gen_two_body_dataset();
    patches = Infer.get_sky_patches(images, two_body; radius_override_pix=50);
    Infer.load_active_pixels!(
        images, patches, noise_fraction=Inf, min_radius_pix=Nullable(10));
    ea_fft = DeterministicVIImagePSF.ElboArgs(
        images, deepcopy(ea.vp), patches, collect(1:ea.S), psf_K=1);
    n = 3

    img_1 = CelesteEDA.render_sources(ea, [1], n, include_epsilon=false, include_iota=false);
    img_2 = CelesteEDA.render_sources(ea, [2], n, include_epsilon=false, include_iota=false);
    img_12 = CelesteEDA.render_sources(ea, [1, 2], n, include_epsilon=false, include_iota=false);
    img_1[isnan.(img_1)] = 0
    img_2[isnan.(img_2)] = 0
    img_12[isnan.(img_12)] = 0
    @test isapprox(maximum(abs.(img_12 - img_1 - img_2)), 0.0, atol = 1e-12)

    ea_fft = DeterministicVIImagePSF.ElboArgs(
        images, deepcopy(ea.vp), patches, collect(1:ea.S), psf_K=1);
    psf_image_mat = Matrix{Float64}[
        PSF.get_psf_at_point(patches[s, b].psf) for s in 1:ea_fft.S, b in 1:ea_fft.N];
    fsm_mat = DeterministicVIImagePSF.FSMSensitiveFloatMatrices[
        DeterministicVIImagePSF.FSMSensitiveFloatMatrices() for
        s in 1:ea_fft.S, b in 1:ea_fft.N];
    DeterministicVIImagePSF.initialize_fsm_sf_matrices!(fsm_mat, ea_fft, psf_image_mat);

    fft_img_1 = CelesteEDA.render_sources_fft(ea_fft, fsm_mat, [1], n, include_epsilon=false, include_iota=false);
    fft_img_2 = CelesteEDA.render_sources_fft(ea_fft, fsm_mat, [2], n, include_epsilon=false, include_iota=false);
    fft_img_12 = CelesteEDA.render_sources_fft(ea_fft, fsm_mat, [1, 2], n, include_epsilon=false, include_iota=false);
    fft_img_1[isnan.(fft_img_1)] = 0
    fft_img_2[isnan.(fft_img_2)] = 0
    fft_img_12[isnan.(fft_img_12)] = 0
    @test isapprox(maximum(abs.(fft_img_12 - fft_img_1 - fft_img_2)), 0.0, atol = 1e-12)

    full_img_12 = CelesteEDA.render_sources(ea, [1, 2], n, include_epsilon=true, include_iota=true);
    full_fft_img_12 = CelesteEDA.render_sources_fft(ea_fft, fsm_mat, [1, 2], n, include_epsilon=true, include_iota=true);
    orig_img_12 = CelesteEDA.show_sources_image(ea, [1, 2], n);
    full_img_12[isnan.(full_img_12)] = 0
    full_fft_img_12[isnan.(full_fft_img_12)] = 0
    orig_img_12[isnan.(orig_img_12)] = 0

    # Check that the two images are roughly the same, and roughly the same as the original..
    @test median(abs.(full_fft_img_12 - full_img_12)) / maximum(abs.(full_img_12)) < 0.01
    @test median(abs.(full_fft_img_12 - orig_img_12)) / maximum(abs.(orig_img_12)) < 0.02
end


test_render()
