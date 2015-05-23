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
#matshow(psf)

# no no no
#gmm = GMM(2, psf; method=:kmeans, kind=:diag, nInit=50, nIter=50, nFinal=50)
band_gain, band_dark_variance = SDSS.load_photo_field(field_dir, run_num, camcol_num, frame_num)

b = 1
nelec, calib_col, sky_grid = SDSS.load_raw_field(field_dir, run_num, camcol_num, frame_num, b, band_gain[b]);

# Get the masked image.
masked_nelec = deepcopy(nelec);
sum(isnan(masked_nelec))
SDSS.mask_image!(masked_nelec, field_dir, run_num, camcol_num, frame_num, mask_planes=Set({"S_MASK_SATUR"}));
sum(isnan(masked_nelec))


fpm_filename = "$field_dir/fpM-$run_num-r$camcol_num-$frame_num.fit"
fpm_fits = FITS(fpm_filename)

# The last header contains the mask.
fpm_mask = fpm_fits[12]
fpm_hdu_indices = read(fpm_mask, "value")
    
mask_types = read(fpm_mask, "attributeName")
plane_rows = findin(mask_types[masktype_rows], mask_planes)

for fpm_i in 1:length(mask_types)
	mask_img = deepcopy(nelec);

	println(mask_types[fpm_i])
    # You want the HDU in 2 + fpm_mask.value[i] for i in keep_rows (in a 1-indexed language).
    mask_index = 2 + fpm_hdu_indices[fpm_i]
    cmin = read(fpm_fits[mask_index], "cmin")
    cmax = read(fpm_fits[mask_index], "cmax")
    rmin = read(fpm_fits[mask_index], "rmin")
    rmax = read(fpm_fits[mask_index], "rmax")
    row0 = read(fpm_fits[mask_index], "row0")
    col0 = read(fpm_fits[mask_index], "col0")

    @assert all(col0 .== 0)
    @assert all(row0 .== 0)
    @assert length(rmin) == length(cmin) == length(rmax) == length(cmax)

    nan_pixels = 0
    for block in 1:length(rmin)
        # The ranges are for a 0-indexed language.
        @assert cmax[block] + 1 <= size(mask_img)[1]
        @assert cmin[block] + 1 >= 1
        @assert rmax[block] + 1 <= size(mask_img)[2]
        @assert rmin[block] + 1 >= 1

        # For some reason, the sizes are inconsistent if the rows are read first.
        # I presume that either these names are strange or I am supposed to read
        # the image from the frame and transpose it.
        mask_img[(cmin[block]:cmax[block]) + 1, (rmin[block]:rmax[block]) + 1] = NaN
        nan_pixels = nan_pixels + (cmax[block] - cmin[block] + 1) * (rmax[block] - rmin[block] + 1)
    end
    println(mask_index, " ", sum(isnan(mask_img)), " ", nan_pixels)
end



#################
# Load python data for comparision.
nelec_py_df = readtable("/tmp/test_3900_6_269_1_img.csv", header=false);
# Note the transpose.  Isn't there a more efficient way to do this?
nelec_py = band_gain[b] * Float64[ nelec_py_df[i, j] for j=1:ncol(nelec_py_df), i=1:nrow(nelec_py_df) ];

masked_nelec_py_df = readtable("/tmp/test_3900_6_269_1_masked_img.csv", header=false);
masked_nelec_py = band_gain[b] * Float64[ masked_nelec_py_df[i, j] for
                                          j=1:ncol(masked_nelec_py_df), i=1:nrow(masked_nelec_py_df) ];

sum(isnan(masked_nelec_py))
sum(isnan(masked_nelec))

tmp = findn(isnan(masked_nelec))
julia_masked = hcat(tmp[1][:], tmp[2][:])

tmp = findn(isnan(masked_nelec_py))
py_masked = hcat(tmp[1][:], tmp[2][:])

julia_coord = Array(ASCIIString, size(julia_masked, 1));
for i = 1:size(julia_masked, 1)
	a = julia_masked[i, 1]
	b = julia_masked[i, 2]
	julia_coord[i] = "$a+$b"
end

py_coord = Array(ASCIIString, size(py_masked, 1));
for i = 1:size(py_masked, 1)
	a = py_masked[i, 1]
	b = py_masked[i, 2]
	py_coord[i] = "$a+$b"
end
setdiff(Set(py_coord), Set(julia_coord))


nelec_py[1:10, 1:10]
nelec[1:10, 1:10]

# This is reasonably close.
nelec_py[1:10, 1:10] ./ nelec[1:10, 1:10]
nelec_py[1:10, 1:10] - nelec[1:10, 1:10]

maximum(abs(nelec_py - nelec))

which(abs(nelec_py - nelec) > 10.)

# This is totally strange.  Python appears to do no processing on the FpC file before turning
# it into an image.
fpc_file = "/home/rgiordan/Documents/git_repos/tractor/fpC-003900-u6-0269.fit"
fpc_fits = FITS(fpc_file)
raw_fpc = convert(Array{Float64,2}, read(fpc_fits[1]));



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

