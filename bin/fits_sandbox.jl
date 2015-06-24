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


#######################
# WCS stuff.  See
# https://github.com/JuliaAstro/FITSIO.jl/issues/39

b = 1
b_letter = band_letters[b]
@assert 1 <= b <= 5
b_letter = band_letters[b]

# A bug in my FITSIO change:
img_filename = "$field_dir/frame-$b_letter-$run_num-$camcol_num-$field_num.fits"
img_fits = FITSIO.FITS(img_filename)
ctype = [FITSIO.read_key(img_fits[1], "CTYPE1")[1], FITSIO.read_key(img_fits[1], "CTYPE2")[1]]
foo = read_header(img_fits[1]);
bar = read_header(img_fits[2]);
header_str = FITSIO.read_header(img_fits[1], ASCIIString)
close(img_fits)

world_coords = Array(Float64, 2, nrow(cat_df))
for n=1:nrow(cat_df)
	world_coords[1, n] = cat_df[n, :ra]
	world_coords[2, n] = cat_df[n, :dec]
end
H = blob[b].H
W = blob[b].W
pixcrd = Float64[0 0; 0 H; 0 W; H W]'

WCSLIB.wcss2p(blob[b].wcs, world_coords)'

[ unsafe_load(blob[b].wcs.crval, i) for i=1:2 ]
[ unsafe_load(blob[b].wcs.cd, i) for i=1:4 ]


# Other way to read it in?  This doesn't seem to work.
naxis = FITSIO.read_key(img_fits[1], "NAXIS")[1]
@assert naxis == 2

ctype = [FITSIO.read_key(img_fits[1], "CTYPE1")[1], FITSIO.read_key(img_fits[1], "CTYPE2")[1]]
crpix = [FITSIO.read_key(img_fits[1], "CRPIX1")[1], FITSIO.read_key(img_fits[1], "CRPIX2")[1]]
crval = [FITSIO.read_key(img_fits[1], "CRVAL1")[1], FITSIO.read_key(img_fits[1], "CRVAL2")[1]]
cunit = [FITSIO.read_key(img_fits[1], "CUNIT1")[1], FITSIO.read_key(img_fits[1], "CUNIT2")[1]]

cd = [ FITSIO.read_key(img_fits[1], "CD1_1")[1] FITSIO.read_key(img_fits[1], "CD1_2")[1];
       FITSIO.read_key(img_fits[1], "CD2_1")[1] FITSIO.read_key(img_fits[1], "CD2_2")[1] ]

# Oh well, this doesn't work.
w2 = WCSLIB.wcsprm(naxis;
		            cd = cd,
		            ctype = ctype,
		            crpix = crpix,
		            crval = crval)

[ unsafe_load(w2.crval, i) for i=1:2 ]

H = blob[b].H
W = blob[b].W

pixcrd = Float64[0 0; 0 H; 0 W; H W]'

# convert pixel -> world coordinates
world1 = WCSLIB.wcsp2s(w2, pixcrd)
world2 = WCSLIB.wcsp2s(wcs, pixcrd)

# convert world -> pixel coordinates
pixcrd12 = WCSLIB.wcss2p(w2, world1)
pixcrd22 = WCSLIB.wcss2p(wcs, world2)


#########
# Compare the PSF with a python-generated file.
b = 1

H = blob[b].H
W = blob[b].W
psf_point_x = H / 2
psf_point_y = W / 2

rrows, rnrow, rncol, cmat = SDSS.load_psf_data(field_dir, run_num, camcol_num, field_num, b);
raw_psf = PSF.get_psf_at_point(psf_point_x, psf_point_y, rrows, rnrow, rncol, cmat);
maximum(raw_psf)

# ########## Here is my code:
# row = psf_point_x
# col = psf_point_y

# # This is a coordinate transform to keep the polynomial coefficients
# # to a reasonable size.
# const rcs = 0.001

# # rrows' image data is in the first column a flattened form.
# # The second dimension is the number of eigen images, which should
# # match the number of coefficient arrays.
# k_tot = size(rrows)[2]
# @assert k_tot == size(cmat)[3]

# nrow_b = size(cmat)[1]
# ncol_b = size(cmat)[2]

# # From the IDL docs:
# # http://photo.astro.princeton.edu/photoop_doc.html#SDSS_PSF_RECON
# #   acoeff_k = SUM_i{ SUM_j{ (0.001*ROWC)^i * (0.001*COLC)^j * C_k_ij } }
# #   psfimage = SUM_k{ acoeff_k * RROWS_k }

# # Get the weights.  To match python, this must be transposed.  Also,
# # row and col are intended to be zero-indexed.
# coeffs_mat = [ (((row - 1) * rcs) ^ i) * (((col - 1) * rcs) ^ j) for
#                 i=0:(nrow_b - 1), j=0:(ncol_b - 1)]
# weight_mat = zeros(k_tot)
# #weight_mat = zeros(nrow_b, ncol_b)
# # NOTE: was previously 1:3, not 1:k_tot, don't know why.
# for k = 1:k_tot, i = 1:nrow_b, j = 1:ncol_b
#     weight_mat[k] += cmat[i, j, k] * coeffs_mat[i, j]
# end

# # Weight the images in rrows and reshape them into matrix form.
# # It seems I need to convert for reshape to work.  :(


# foo = 1300
# raw_psfs = [ rrows[:, i] * weight_mat[i] for i=1:k_tot];
# # raw_psf_sum = reduce(sum, raw_psfs) # Doesn't work!
# raw_psf_sum = sum(raw_psfs)

# raw_psfs[1][foo] + raw_psfs[2][foo] + raw_psfs[3][foo] + raw_psfs[4][foo] 
# sum([ raw_psfs[k][foo] for k=1:k_tot])
# raw_psf_sum[foo]

# raw_psf = reshape(sum(raw_psfs),
#                   (convert(Int64, rnrow), convert(Int64, rncol)));

# raw_psf_t = reshape(reduce(sum, [ rrows[:, i] * weight_mat[i] for i=1:k_tot]),
#                     (convert(Int64, rncol), convert(Int64, rnrow)));

# sp1 = PyPlot.matshow(raw_psf)
# PyPlot.colorbar(sp1)

#writedlm("/tmp/julia_psf.csv", raw_psf, ',')
py_raw_psf = readdlm("/tmp/py_psf.csv", ',')

close()
plot(py_raw_psf, raw_psf', "k.")
plot(py_raw_psf, py_raw_psf, "r.")
