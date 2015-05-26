using Celeste
using CelesteTypes

using DataFrames
using SampleData

using SDSS
import PSF

import PyPlot

# Some examples of the SDSS fits functions.

# Blob for comparison
blob, mp, one_body = gen_sample_galaxy_dataset();

field_dir = joinpath(dat_dir, "sample_field")
run_num = "003900"
camcol_num = "6"
frame_num = "0269"


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


# Load the catalog entry for a field.

# http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/RERUN/RUN/CAMCOL/photoObj.html
# In python:
# 
# from tractor.sdss import *
# sdss = DR8()
# sdss.get_url('photoObj', args.run, args.camcol, args.field)
#
# See tractor/sdss.py:_get_sources
# https://github.com/dstndstn/tractor/blob/f1d92f0569e61f920932635b469222a2ac989ed7/tractor/sdss.py#L217

cat_filename = "$field_dir/photoObj-$run_num-$camcol_num-$frame_num.fits"
cat_fits = FITS(cat_filename)

# The second block has the objects, e.g.:
read(cat_fits[2], "PHI_DEV_DEG")
read(cat_fits[2], "RESOLVE_STATUS") # some kind of bitmap?
unique(read(cat_fits[2], "OBJC_TYPE"))