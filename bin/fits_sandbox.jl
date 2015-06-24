using Celeste
using CelesteTypes

using DataFrames
using SampleData

import SDSS
import PSF
import FITSIO
import PyPlot

# Some examples of the SDSS fits functions.
field_dir = joinpath(dat_dir, "sample_field")
run_num = "003900"
camcol_num = "6"
field_num = "0269"

const band_letters = ['u', 'g', 'r', 'i', 'z']

b = 1
b_letter = band_letters[b]

#############
# Load the catalog

# In python:
# 
# from tractor.sdss import *
# sdss = DR8()
# sdss.get_url('photoObj', args.run, args.camcol, args.field)

blob = SDSS.load_sdss_blob(field_dir, run_num, camcol_num, field_num);
cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);
cat_entries = SDSS.convert_catalog_to_celeste(cat_df, blob);
cat_coords = convert(Array{Float64}, cat_df[[:ra, :dec]])'


###############
# Write the data to csv.

# Write pixel coordinates to the catalog
for b=1:5
	pix_coords = WCSLIB.wcss2p(blob[b].wcs, cat_coords)
	cat_df[symbol("pix_x_$(b)")] = pix_coords[1,:][:]
	cat_df[symbol("pix_y_$(b)")] = pix_coords[2,:][:]
end
writetable("/tmp/catalog-$run_num-$camcol_num-$field_num.csv", cat_df)

# Write to csv for easy viewing.
for b=1:5
	b_letter = band_letters[b]
	writedlm("/tmp/frame-$b_letter-$run_num-$camcol_num-$field_num.csv", blob[b].pixels, ',')
end



#########
# Compare the PSF with a python-generated file.
b = 1

H = blob[b].H
W = blob[b].W
psf_point_x = H / 2
psf_point_y = W / 2

rrows, rnrow, rncol, cmat = SDSS.load_psf_data(field_dir, run_num, camcol_num, field_num, b);
raw_psf = PSF.get_psf_at_point(psf_point_x, psf_point_y, rrows, rnrow, rncol, cmat);

# This has to be generated by hand using get_astrometry_psf.py, and should be
# based on the same file as used above.  All the python indices are zero-based.
# Example command:
# python bin/get_astrometry_psf.py --ps_file=/tmp/psField_003900_6_0269.fit \
#     --band=0 --row=1023 --col=743.5 --destination_file=/tmp/py_psf.csv
py_raw_psf = readdlm("/tmp/py_psf.csv", ',');

close()
PyPlot.plot(py_raw_psf, raw_psf', "k.");
PyPlot.plot(py_raw_psf, py_raw_psf, "r.");




#######################
# Checking the WCS stuff.

world_coords = Array(Float64, 2, nrow(cat_df))
for n=1:nrow(cat_df)
	world_coords[1, n] = cat_df[n, :ra]
	world_coords[2, n] = cat_df[n, :dec]
end
H = blob[b].H
W = blob[b].W
pixcrd = Float64[0 0; 0 H; 0 W; H W]'

# convert pixel -> world coordinates and back.
world = WCSLIB.wcsp2s(blob[b].wcs, pixcrd)
pixcrd2 = WCSLIB.wcss2p(blob[b].wcs, world)

# Read in the catalog pixel coordinates:
WCSLIB.wcss2p(blob[b].wcs, world_coords)'

# Here's how to read WCS attributes:
[ unsafe_load(blob[b].wcs.crval, i) for i=1:2 ]
[ unsafe_load(blob[b].wcs.cd, i) for i=1:4 ]


# Can we read it in directly?  This doesn't seem to work.
naxis = FITSIO.read_key(img_fits[1], "NAXIS")[1]
@assert naxis == 2

ctype = [FITSIO.read_key(img_fits[1], "CTYPE1")[1], FITSIO.read_key(img_fits[1], "CTYPE2")[1]]
crpix = [FITSIO.read_key(img_fits[1], "CRPIX1")[1], FITSIO.read_key(img_fits[1], "CRPIX2")[1]]
crval = [FITSIO.read_key(img_fits[1], "CRVAL1")[1], FITSIO.read_key(img_fits[1], "CRVAL2")[1]]
cunit = [FITSIO.read_key(img_fits[1], "CUNIT1")[1], FITSIO.read_key(img_fits[1], "CUNIT2")[1]]

cd = [ FITSIO.read_key(img_fits[1], "CD1_1")[1] FITSIO.read_key(img_fits[1], "CD1_2")[1];
       FITSIO.read_key(img_fits[1], "CD2_1")[1] FITSIO.read_key(img_fits[1], "CD2_2")[1] ]

w2 = WCSLIB.wcsprm(naxis;
		            cd = cd,
		            ctype = ctype,
		            crpix = crpix,
		            crval = crval)

[ unsafe_load(w2.crval, i) for i=1:2 ]


