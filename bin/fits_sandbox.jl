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

img_filename = "$field_dir/frame-$b_letter-$run_num-$camcol_num-$field_num.fits"
img_fits = FITSIO.FITS(img_filename)
@assert length(img_fits) == 4

FITSIO.read_header(img_fits[1], ASCIIString)

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


# Write upt for easy viewing.
for b=1:5
	b_letter = band_letters[b]
	writedlm("/tmp/frame-$b_letter-$run_num-$camcol_num-$field_num.csv", blob[b].pixels, ',')
end
writetable("/tmp/catalog-$b_letter-$run_num-$camcol_num-$field_num.csv", cat_df)


#######################
# WCS stuff.  See
# https://github.com/JuliaAstro/FITSIO.jl/issues/39

b_letter = band_letters[b]

img_filename = "$field_dir/frame-$b_letter-$run_num-$camcol_num-$field_num.fits"
img_fits = FITSIO.FITS(img_filename)
@assert length(img_fits) == 4
header_str = FITSIO.read_header(img_fits[1], ASCIIString)
((wcs,),nrejected) = WCSLIB.wcspih(header_str)

world_coords = Array(Float64, 2, nrow(cat_df))
for n=1:nrow(cat_df)
	world_coords[2, n] = cat_df[n, :ra]
	world_coords[1, n] = cat_df[n, :dec]
end

pixel_coords = WCSLIB.wcss2p(blob[1].wcs, world_coords)

# Other way to read it in?
naxis = FITSIO.read_key(img_fits[1], "NAXIS")[1]
@assert naxis == 2

# cdelt = [FITSIO.read_key(img_fits[1], "NAXIS")[1]; FITSIO.read_key(img_fits[1], "NAXIS")[1]]
ctype = [FITSIO.read_key(img_fits[1], "CTYPE1")[1], FITSIO.read_key(img_fits[1], "CTYPE2")[1]]
crpix = [FITSIO.read_key(img_fits[1], "CRPIX1")[1], FITSIO.read_key(img_fits[1], "CRPIX2")[1]]
crval = [FITSIO.read_key(img_fits[1], "CRVAL1")[1], FITSIO.read_key(img_fits[1], "CRVAL2")[1]]

# I don't know what the rest of these things should be, and the others are just a guess.
w = wcsprm(naxis; 
           cdelt = [-0.066667, 0.066667],
           ctype = ctype,
           crpix = crpix,
           crval = val,
           pv    = [pvcard(2, 1, 45.0)])
