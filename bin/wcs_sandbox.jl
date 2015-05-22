using Celeste
using CelesteTypes
using PyPlot

using FITSIO
using WCSLIB
using DataFrames
using SampleData

using Grid
using MixtureModels

using SDSS

# Blob for comparison

blob, mp, one_body = gen_sample_galaxy_dataset();

field_dir = joinpath(dat_dir, "sample_field")
run_num = "003900"
camcol_num = "6"
frame_num = "0269"


rrows, rnrow, rncol, cmat = SDSS.load_psf_data(field_dir, run_num, camcol_num, frame_num, 1);
psf = SDSS.get_psf_at_point(1., 1., rrows, rnrow, rncol, cmat);
matshow(psf)

# no no no
#gmm = GMM(2, psf; method=:kmeans, kind=:diag, nInit=50, nIter=50, nFinal=50)
band_gain, band_dark_variance = SDSS.load_photo_field(field_dir, run_num, camcol_num, frame_num)

b = 1
nelec, calib_col, sky_grid = SDSS.load_raw_field(field_dir, run_num, camcol_num, frame_num, b, band_gain[b]);
masked_nelec = deepcopy(nelec);
SDSS.mask_image!(masked_nelec, field_dir, run_num, camcol_num, frame_num);

nelec_py_df = readtable("/tmp/test_3900_6_269_1_img.csv", header=false);
nelec_py = [ nelec_py_df[i, j] for i=1:nrow(nelec_py_df), j=1:ncol(nelec_py_df)];



#######################
# Get the PSF
# "Documented" here:
# http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/psField.html

# I think you need to replicate the code in 
# https://github.com/rgiordan/astrometry.net/blob/master/sdss/common.py
# ... called in 
# https://github.com/dstndstn/astrometry.net/blob/master/sdss/dr7.py
# There is also this, which is a bit more explicit:
# https://www.sdss3.org/dr8/algorithms/read_psf.php
# Or the dr12 version:
# http://www.sdss.org/dr12/algorithms/read_psf/

# File is from:
#  'http://data.sdss3.org/sas/dr10/boss/photo/redux/301/3900/objcs/6/psField-003900-6-0269.fit'

# For reference, you can get the URLs in python with
# >>> from astrometry.sdss import *
# >>> sdss = DR10()
# >>> sdss.get_url('psField', 3900, 6, 269, 'r')
# 'http://data.sdss3.org/sas/dr10/boss/photo/redux/301/3900/objcs/6/psField-003900-6-0269.fit'



###############################
# calibration
# Data described here:
# http://www.sdss.org/dr12/data_access/bulk/
#
# Data type is perhaps
# http://data.sdss3.org/datamodel/files/PHOTO_SWEEP/RERUN/calibObj.html
#
# Which is maybe here:
# http://data.sdss3.org/sas/dr12/boss/sweeps/dr9/301/calibObj-003900-6-sky.fits.gz
# http://data.sdss3.org/sas/dr12/boss/sweeps/dr9/301/calibObj-003900-6-gal.fits.gz
# http://data.sdss3.org/sas/dr12/boss/sweeps/dr9/301/calibObj-003900-6-star.fits.gz
# ...but these files are huge.  Why?

