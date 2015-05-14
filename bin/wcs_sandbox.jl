using Celeste
using CelesteTypes
#using Gadfly

using FITSIO
using WCSLIB
using DataFrames
using SampleData

using FITSIO

using DataFrames
using Grid

# Blob for comparison

blob, mp, one_body = gen_sample_galaxy_dataset();



field_dir = joinpath(dat_dir, "sample_field")
run_num = "003900"
camcol_num = "6"
frame_num = "0269"


# This is the calibration (I think)?
# http://data.sdss3.org/datamodel/files/PHOTO_CALIB/RERUN/RUN/nfcalib/calibPhotomGlobal.html



# Read the catalog entry (?)
photofield_filename = "$field_dir/photoField-$run_num-$camcol_num.fits"
photofield_fits = FITS(photofield_filename)

# http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/RERUN/RUN/photoField.html
# Keywords in the header:
length(photofield_fits)
read_header(photofield_fits[1])
read_key(photofield_fits[1], "RUN")

# The table.  You can only read one column at a time.
read_fields = ["run", "rerun", "camcol", "skyversion", "field", "nStars"]
df = DataFrame()
for field in read_fields
	println(field)
	this_col = collect(read(photofield_fits[2], field))
	println(size(this_col))
    df[DataFrames.identifier(field)] = this_col;
end

field_row = read(photofield_fits[2], "field") .== int(frame_num);
band_gain = read(photofield_fits[2], "gain");
band_dark_variance = collect(read(photofield_fits[2], "dark_variance")[:, field_row]);

df[df[:field] .== int(frame_num), :]

# Read the image data.
# Documented here:
# http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html

# This is the sky bacgkround:
sky_image_raw = read(img_fits[3], "ALLSKY");
sky_x = collect(read(img_fits[3], "XINTERP"));
sky_y = collect(read(img_fits[3], "YINTERP"));

# Combining the example from
# http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
# ...with the documentation from the IDL language:
# http://www.exelisvis.com/docs/INTERPOLATE.html
# ...we can see that we are supposed to interpret a point (sky_x[i], sky_y[j])
# with associated row and column (if, jf) = (floor(sky_x[i]), floor(sky_y[j]))
# as lying in the square spanned by the points
# (sky_image_raw[if, jf], sky_image_raw[if + 1, jf + 1]).
# ...keeping in mind that IDL uses zero indexing:
# http://www.exelisvis.com/docs/Manipulating_Arrays.html

sky_grid_vals = ((1:1.:size(sky_image_raw)[1]) - 1, (1:1.:size(sky_image_raw)[2]) - 1);
sky_grid = CoordInterpGrid(sky_grid_vals, sky_image_raw[:,:,1], BCnearest, InterpLinear);
sky_image = [ sky_grid[x, y] for x in sky_x, y in sky_y ];

# This is the calibration vector:
calib_row = read(img_fits[2]);
calib_image = [ calib_row[x] for x in 1:size(processed_image)[1], y in 1:size(processed_image)[2] ];


b = 3
b_letter = ['u', 'g', 'r', 'i', 'z'][b]

img_filename = "$field_dir/frame-$b_letter-$run_num-$camcol_num-$frame_num.fits"
img_fits = FITS(img_filename)
length(img_fits) # Should be 4

# This is the sky-subtracted and calibrated image.  There are no fields in the first header.
processed_image = read(img_fits[1]);

# Convert to raw electron counts.
dn = convert(Array{Float64, 2}, (processed_image ./ calib_image .+ sky_image));
n_elec = band_gain[b] * dn;

# Why aren't these at least approximately integers?
n_elec[1:10, 1:10]

# This is supposed to be the error in nanomaggies, if you care.  Why dn / gain?
dn_err = sqrt(dn / band_gain[b] + band_dark_variance[b]);

# Apparently dn * cimg is in units of nanomaggies:
#
#dn= img/cimg+simg
#nelec= dn*gain
# var(nelec) = dn * gain
# var(dn) = var(nelec / gain) = dn / gain
# sd(dn) = sqrt(dn / gain)
#dn_err = sqrt(dn / gain + darkVariance)
#img_err = dn_err*cimg (nanomaggies)


# Get the PSF
psf_filename = "$field_dir/psField-$run_num-$camcol_num-$frame_num.fit";
psf_fits = FITS(psf_filename);

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

#psf_filename = "psField-003900-6-0269.fit"
psf_fits = FITS(psf_filename);
b = 3
read_header(psf_fits[b + 1])
# ...other fields...
# TFORM7  =                 '1J' / PIXDATATYPE
# TTYPE7  =              'RTYPE' / type
# TFORM8  =             '1PE(0)' / FLOAT
# TTYPE8  =              'RROWS' / rows_fl32
# TFORM9  =                 '1J' / INT
# TTYPE9  =              'RROW0' / row0
# TFORM10 =                 '1J' / INT
# TTYPE10 =              'RCOL0' / col0
# ....

# reconstruct the PSF at location (row,col):
foo = read_header(psf_fits[b + 1])
rrows = read(psf_fits[b + 1], "RROWS")

read(psf_fits[b + 1], "RNROW")
# 4-element Array{Int32,1}:
#  51
#  51
#  51
#  51

read(psf_fits[b + 1], "RROWS")
# ERROR: key not found: -42
#  in read at /home/rgiordan/.julia/v0.3/FITSIO/src/hdutypes.jl:72


#nrow_b=(pstruct.nrow_b)[0]
nrow_b = read(psf_fits[b + 1], "nrow_b")[1]

#ncol_b=(pstruct.ncol_b)[0]
ncol_b = read(psf_fits[b + 1], "ncol_b")[1]

#;assumes they are the same for each eigen so only use the 0 one
#rnrow=(pstruct.rnrow)[0]
#rncol=(pstruct.rncol)[0]
rnrow = read(psf_fits[b + 1], "rnrow")[1]
rncol = read(psf_fits[b + 1], "rncol")[1]

nb = nrow_b * ncol_b

#coeffs=fltarr(nb)
coeffs = zeros(Float64, nb)

#ecoeff=fltarr(3)
ecoeff = zeros(Float64, 3)

#cmat=pstruct.c
cmat = read(psf_fits[b + 1], "c")

rcs = 0.001
#FOR i=0L, nb-1L DO coeffs[i]=(row*rcs)^(i mod nrow_b) * (col*rcs)^(i/nrow_b)
coeffs = [ (row * rcs)^(i % nrow_b) * (col * rcs)^(i / nrow_b) for i=0:(nb - 1)]
FOR j=0,2 DO BEGIN
    FOR i=0L, nb-1L DO BEGIN
        ecoeff[j]=ecoeff[j]+cmat(i/nrow_b,i mod nrow_b,j)*coeffs[i]
    ENDFOR
ENDFOR
psf = (pstruct.rrows)[*,0]*ecoeff[0]+$
      (pstruct.rrows)[*,1]*ecoeff[1]+$
      (pstruct.rrows)[*,2]*ecoeff[2]





# From the frame reference.  Do we need to do this?  Yes.
## Finally, there are some areas of the image which are part of bleed trails, bad columns, and the like.
## If you require to track those in your analysis (e.g. weight them at zero) then you need to use the
## fpM files. Those files are in a special format, best read using the stand-alone atlas reader
## software. Use the utility called read_mask.

# Copying Dustin's code from
# https://github.com/dstndstn/astrometry.net/blob/master/sdss/common.py

# http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/fpM.html
fpm_filename = "$field_dir/fpM-$run_num-r$camcol_num-$frame_num.fit";
fpm_fits = FITS(fpm_filename);

# The last header contains the mask.
fpm_mask = fpm_fits[12]

# TODO: only check these rows:
masktype_rows = read(fpm_mask, "defName") .== "S_MASKTYPE"

# From sdss/dr8.py:
#   for plane in [ 'INTERP', 'SATUR', 'CR', 'GHOST' ]:
#            fpM.setMaskedPixels(plane, invvar, 0, roi=roi)
# What is the meaning of these?
# Apparently attributeName lists the meanings of the HDUs in order.
keep_planes = Set({"S_MASK_INTERP", "S_MASK_SATUR", "S_MASK_CR", "S_MASK_GHOST"});
keep_rows = [ read(fpm_mask, "attributeName")[i] in keep_planes for i=1:11 ]
# You want the HDU in 2 + fpm_mask.value[i] for i in keep_rows (in a 1-indexed language).
read(fpm_mask, "value")

# Then read the rectangles from fpm_fits[whatever], which contains an array of row and column minima.