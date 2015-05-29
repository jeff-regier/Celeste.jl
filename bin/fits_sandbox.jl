using Celeste
using CelesteTypes

using DataFrames
using SampleData

using SDSS
import PSF
import FITSIO
import PyPlot

# Some examples of the SDSS fits functions.

# Blob for comparison
blob, mp, one_body = gen_sample_galaxy_dataset();

field_dir = joinpath(dat_dir, "sample_field")
run_num = "003900"
camcol_num = "6"
frame_num = "0269"

#############
# Load the catalog

# In python:
# 
# from tractor.sdss import *
# sdss = DR8()
# sdss.get_url('photoObj', args.run, args.camcol, args.field)

blob = SDSS.load_stamp_blob(SampleData.dat_dir, "5.0073-0.0739");
stamp_cat_entries_df =
	SDSS.load_stamp_catalog_df(SampleData.dat_dir, "s82-5.0073-0.0739", blob);
stamp_catalog = SDSS.load_stamp_catalog(SampleData.dat_dir, "s82-5.0073-0.0739", blob);

cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, frame_num);
# Not the right blob, just to test:
cat_entries = SDSS.convert_catalog_to_celeste(cat_df, blob)

#######################
# Load the image data
band_gain, band_dark_variance = SDSS.load_photo_field(field_dir, run_num, camcol_num, frame_num)

b = 1
nelec, calib_col, sky_grid = SDSS.load_raw_field(field_dir, run_num, camcol_num, frame_num, b, band_gain[b]);

# Get the masked image with and without the (probably buggy) python indexing.
masked_nelec_bad = deepcopy(nelec);
SDSS.mask_image!(masked_nelec_bad, field_dir, run_num, camcol_num, frame_num, b, python_indexing=true);
sum(isnan(masked_nelec_bad))

masked_nelec = deepcopy(nelec);
SDSS.mask_image!(masked_nelec, field_dir, run_num, camcol_num, frame_num, b);
sum(isnan(masked_nelec))


###############
# Load and fit a mixture of Gaussians to the psf

rrows, rnrow, rncol, cmat = SDSS.load_psf_data(field_dir, run_num, camcol_num, frame_num, 1);

psf = PSF.get_psf_at_point(1., 1., rrows, rnrow, rncol, cmat);
gmm = PSF.fit_psf_gaussians(psf);

gmm_fit = sum(psf) * Float64[ PSF.evaluate_gmm(gmm, Float64[x, y]')[1] for x=1:size(psf, 1), y=1:size(psf, 2) ]

if false
	PyPlot.matshow(gmm_fit)
	PyPlot.title("fit")

	PyPlot.matshow(psf)
	PyPlot.title("psf")

	PyPlot.matshow(psf - gmm_fit)
	PyPlot.title("diff")
end

sum((psf - gmm_fit) ^ 2)

PSF.convert_gmm_to_celeste(gmm)
