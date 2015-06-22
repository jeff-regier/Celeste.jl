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

cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);

blob = SDSS.load_sdss_blob(field_dir, run_num, camcol_num, field_num);
cat_entries = SDSS.convert_catalog_to_celeste(cat_df, blob)


#######################
# WCS stuff.  See
# https://github.com/JuliaAstro/FITSIO.jl/issues/39

b_letter = band_letters[b]

img_filename = "$field_dir/frame-$b_letter-$run_num-$camcol_num-$field_num.fits"
img_fits = FITSIO.FITS(img_filename)
@assert length(img_fits) == 4

fits_file = FITSIO.Libcfitsio.fits_open_file(img_filename)
header_str = FITSIO.Libcfitsio.fits_hdr2str(fits_file)
((wcs,),nrejected) = WCSLIB.wcspih(header_str)

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
