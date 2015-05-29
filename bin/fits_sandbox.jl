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




cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, frame_num);



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

# Reading this in:
# type CatalogEntry
#     pos::Vector{Float64}
#     is_star::Bool
#     star_fluxes::Vector{Float64}
#     gal_fluxes::Vector{Float64}
#     gal_frac_dev::Float64
#     gal_ab::Float64
#     gal_angle::Float64
#     gal_scale::Float64
# end

# Is this standard?
bandname = 'r'
bandnum = find(band_letters .== bandname)[1]

cat_filename = "$field_dir/photoObj-$run_num-$camcol_num-$frame_num.fits"
cat_fits = FITSIO.FITS(cat_filename)

cat_hdu = cat_fits[2]

# Eliminate "bright" objects.
# photo_flags1_map is defined in astrometry.net/sdss/common.py
# ...where we see that photo_flags1_map['BRIGHT'] = 2
const bright_bitmask = 2
is_bright = read(cat_hdu, "objc_flags") & bright_bitmask .> 0
has_child = read(cat_hdu, "nchild") .> 0

# What does cutToPrimary mean?  -- we can set it to false

# What is useObjcType?  -- we should use it in stead of prob_psf
objc_type = read(cat_hdu, "objc_type");
[ (x, sum(objc_type .== x)) for x = unique(objc_type) ]

# What is type 8?

# What does it mean for this to be -9999?
raw_prob_psf = read(cat_hdu, "prob_psf");

prob_psf = read(cat_hdu, "prob_psf")[bandnum, :][:];
[ (x, sum(prob_psf .== x)) for x=unique(prob_psf) ]
prob_type = [ (objc_type[i], prob_psf[i]) for i=1:size(prob_psf, 1)] ;
[ (x, sum(prob_type .== x)) for x=unique(prob_type) ]

# TODO: don't do a float comparision?
# is_star = prob_psf .== 1
# is_gal = prob_psf .== 0
# is_bad_psf = (prob_psf .< 0.) | (prob_psf .> 1.)
is_star = objc_type .== 6
is_gal = objc_type .== 3
is_bad_obj = !(is_star | is_gal)

# Unlike the python, record galaxy estimates even when l_gal = 0.
# Is this the right thing to do?
fracdev = read(cat_hdu, "fracdev")[bandnum, :][:];
has_dev = fracdev .> 0.
has_exp = fracdev .< 1.
is_comp = has_dev & has_exp
is_bad_fracdev = (fracdev .< 0.) | (fracdev .> 1)

# WTF?  I believe we skip uncertain objects of ambiguous type.
any(is_comp & is_star)
any((!is_star) & (!is_gal))

# What units are the star fluxes supposed to be in?
starflux = read(cat_hdu, "psfflux")

# Which of these should we record?
compflux = read(cat_hdu, "cmodelflux")
devflux = read(cat_hdu, "devflux")
expflux = read(cat_hdu, "expflux")

galflux = Array(Float64, size(compflux))
for i=1:size(gal_flux, 2)
	if is_comp[i]
		galflux[:, i] = compflux[:, i]
	elseif has_dev[i]
		galflux[:, i] = devflux[:, i]
		@assert !has_exp[i]
	elseif has_exp[i]
		galflux[:, i] = expflux[:, i]
	else
		error("Galaxy has no known type.")
	end
end

# What does fixedComposites mean? (defaults to false)

# This is strange.  I thought fracdev would be the probability of
# being a dev type galaxy and compflux would reflect that, but 
# it does not:
#
# julia> hcat(compflux[1,:]', devflux[1,:]', expflux[1,:]', fracdev)
# 981x4 Array{Float32,2}:
#   compflux      devflux       expflux          fracdev
#   910.639       910.734       910.639          0.863811
#   907.784       907.902       907.784          0.859879
#   907.807       907.931       907.807          0.857916
#     0.667391      0.667391      0.30009        0.040005
#     0.524619      0.524619      0.469452       0.0     
#    -0.0597614    -0.059767     -0.0597614      0.0     
#    18.1697       18.1697        9.6434         0.0     
#  1716.65       1716.66       1716.57           0.79117

ra = read(cat_hdu, "ra")
dec = read(cat_hdu, "dec")

# This is the "position".  In our catalog entry, what is it exactly?
# RaDecPos(ra, dec)

# Should we really multiply by -1?  Maybe no.
# phi_dev_deg = read(cat_hdu, "phi_dev_deg") * -1
# phi_exp_deg = read(cat_hdu, "phi_exp_deg") * -1
phi_dev_deg = read(cat_hdu, "phi_dev_deg")
phi_exp_deg = read(cat_hdu, "phi_exp_deg")

# What is the meaning of this comment?  Dustin does not remember.
# MAGIC -- minimum size of galaxy.
# theta_dev = max(read(cat_hdu, "theta_dev"), 1./30.)
# theta_exp = max(read(cat_hdu, "theta_exp"), 1./30.)
theta_dev = read(cat_hdu, "theta_dev")
theta_exp = read(cat_hdu, "theta_exp")

ab_exp = read(cat_hdu, "ab_exp")
re_exp = theta_exp
phi_exp = phi_exp_deg

ab_dev = read(cat_hdu, "ab_dev")
re_dev = theta_dev
phi_dev = phi_dev_deg


# Skip "bright" rows and those with prob_psf not in [0, 1].
is_bad = is_bad_fracdev | is_bad_obj | is_bright | has_child

objid = read(cat_hdu, "objid");

sum(is_bad) / length(rows)
catalog = Array(CatalogEntry, sum(!is_bad));

cat_df = DataFrame(objid=objid, ra=ra, dec=dec, is_star=is_star, is_gal=is_gal, fracdev=fracdev,
	               theta_exp=theta_exp, ab_exp=ab_exp, re_exp=re_exp, phi_exp=phi_exp,
	               theta_dev=theta_dev, ab_dev=ab_dev, re_dev=re_dev, phi_dev=phi_dev)

for bandnum=1:length(band_letters)
	cat_df[symbol(string("star_flux_", bandnum))] = starflux[bandnum, :][:]
	cat_df[symbol(string("gal_flux_", bandnum))] = galflux[bandnum, :][:]
end

cat_df = cat_df[!is_bad, :]



# Note that I don't think we need / use the brightness
# values, so don't need flux2bright = nmgy2bright


