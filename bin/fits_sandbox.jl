using Celeste
using CelesteTypes

using FITSIO
using WCSLIB
using DataFrames
using SampleData

using Grid
using GaussianMixtures
using Distributions

using SDSS

import PyPlot


# Blob for comparison

blob, mp, one_body = gen_sample_galaxy_dataset();

field_dir = joinpath(dat_dir, "sample_field")
run_num = "003900"
camcol_num = "6"
frame_num = "0269"

rrows, rnrow, rncol, cmat = SDSS.load_psf_data(field_dir, run_num, camcol_num, frame_num, 1);
psf = SDSS.get_psf_at_point(1., 1., rrows, rnrow, rncol, cmat);
#matshow(psf)

# no no no
#gmm = GMM(2, psf; method=:kmeans, kind=:diag, nInit=50, nIter=50, nFinal=50)
band_gain, band_dark_variance = SDSS.load_photo_field(field_dir, run_num, camcol_num, frame_num)

b = 1
nelec, calib_col, sky_grid = SDSS.load_raw_field(field_dir, run_num, camcol_num, frame_num, b, band_gain[b]);

# Get the masked image.
masked_nelec = deepcopy(nelec);
SDSS.mask_image!(masked_nelec, field_dir, run_num, camcol_num, frame_num, b);
sum(isnan(masked_nelec))

masked_nelec2 = deepcopy(nelec);
SDSS.mask_image!(masked_nelec2, field_dir, run_num, camcol_num, frame_num, b, python_indexing=false);
sum(isnan(masked_nelec2))



###############
# Fit a mixture of Gaussians to the psf


gmm = SDSS.fit_psf_gaussians(psf);

gmm_fit = sum(psf) * Float64[ SDSS.evaluate_gmm(gmm, Float64[x, y]')[1] for x=1:size(psf, 1), y=1:size(psf, 2) ]

PyPlot.matshow(gmm_fit)
PyPlot.title("fit")

PyPlot.matshow(psf)
PyPlot.title("psf")

PyPlot.matshow(psf - gmm_fit)
PyPlot.title("diff")

sum((psf - gmm_fit) ^ 2)

SDSS.convert_gmm_to_celeste(gmm)





#################
# Load python data for comparision.  NB: it is not comparable.


nelec_py_df = readtable("/tmp/test_3900_6_269_1_img.csv", header=false);
# Note the transpose.  Isn't there a more efficient way to do this?
nelec_py = band_gain[b] * Float64[ nelec_py_df[i, j] for j=1:ncol(nelec_py_df), i=1:nrow(nelec_py_df) ];

masked_nelec_py_df = readtable("/tmp/test_3900_6_269_1_masked_img.csv", header=false);
masked_nelec_py = band_gain[b] * Float64[ masked_nelec_py_df[i, j] for
                                          j=1:ncol(masked_nelec_py_df), i=1:nrow(masked_nelec_py_df) ];

sum(isnan(masked_nelec_py))
sum(isnan(masked_nelec))

julia_masked = findn(isnan(masked_nelec));
py_masked = findn(isnan(masked_nelec_py));
julia_masked == py_masked

nelec_py[1:10, 1:10]
nelec[1:10, 1:10]

# This is reasonably close.
nelec_py[1:10, 1:10] ./ nelec[1:10, 1:10]
nelec_py[1:10, 1:10] - nelec[1:10, 1:10]
maximum(abs(nelec_py - nelec))
ratio_err = abs(nelec_py ./ nelec - 1);
maximum(ratio_err)
findn(ratio_err .> 1.0001)

# Get the relative errors with and without python masking.
mask_ratio_err = abs(masked_nelec_py ./ masked_nelec - 1);
mask_ratio_err2 = abs(masked_nelec_py ./ masked_nelec2 - 1);

maximum(mask_ratio_err)
maximum(mask_ratio_err2)

sum(mask_ratio_err .> 0.01) / sum(!isnan(masked_nelec))
sum(mask_ratio_err2 .> 0.01) / sum(!isnan(masked_nelec2))

matshow(mask_ratio_err2 .> 0.01)


#x_range = [ minimum(log(nelec)), maximum(log(nelec)) ]
#plot(linrange(x_range[1], x_range[2], 1000), linrange(x_range[1], x_range[2], 1000), "b.")
log_nelec_py = collect(log(masked_nelec_py));
log_nelec2 = collect(log(masked_nelec2));
log_nelec = collect(log(masked_nelec));

close()
subplot(121)
plot(log_nelec_py, log_nelec, "b.")
title("Pixel discrepancy (python masking)")
xlabel("ln(python pixel values)")
ylabel("ln(julia pixel values)")

subplot(122)
plot(log_nelec_py, log_nelec2, "b.")
title("Pixel discrepancy (julia masking)")
xlabel("ln(python pixel values)")
ylabel("ln(julia pixel values)")
close()


# Python appears to do no processing on the FpC file before turning
# it into an image.  Where do the fpC files come from?
fpc_file = "/home/rgiordan/Documents/git_repos/tractor/fpC-003900-u6-0269.fit"
fpc_fits = FITS(fpc_file)
raw_fpc = convert(Array{Float64,2}, read(fpc_fits[1]));


############
# Look at the PSF

# NB: does not contain a psf.
psf_py_df = readtable("/tmp/test_3900_6_269_1_psf.csv", header=false);
psf_py = Float64[ masked_nelec_py_df[i, j] for
                  j=1:ncol(masked_nelec_py_df), i=1:nrow(masked_nelec_py_df) ];
