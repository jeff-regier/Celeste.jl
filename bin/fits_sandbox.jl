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

# The second block has the objects, e.g.:
read(cat_hdu, "PHI_DEV_DEG")
read(cat_hdu, "RESOLVE_STATUS") # some kind of bitmap?
unique(read(cat_hdu, "OBJC_TYPE"))

# Eliminate "bright" objects.
# photo_flags1_map is defined in astrometry.net/sdss/common.py
# ...where we see that photo_flags1_map['BRIGHT'] = 2
const bright_bitmask = 2
is_bright = read(cat_hdu, "objc_flags") & bright_bitmask .> 0
has_child = read(cat_hdu, "nchild") .> 0

# What does cutToPrimary mean?

# What is useObjcType?

# What does it mean for this to be -9999?
prob_psf = read(cat_hdu, "prob_psf")[bandnum, :][:];
[ (x, sum(prob_psf .== x)) for x=unique(prob_psf) ]

# TODO: don't do a float comparision?
is_star = prob_psf .== 1
is_gal = prob_psf .== 0
is_bad_psf = (prob_psf .< 0.) | (prob_psf .> 1.)

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


# Should we really multiply by -1?
phi_dev_deg = read(cat_hdu, "phi_dev_deg") * -1
phi_exp_deg = read(cat_hdu, "phi_exp_deg") * -1

# What is the meaning of this comment?
# MAGIC -- minimum size of galaxy.
theta_dev = max(read(cat_hdu, "theta_dev"), 1./30.)
theta_exp = max(read(cat_hdu, "theta_exp"), 1./30.)

ab_exp = read(cat_hdu, "ab_exp")
re_exp = theta_exp
phi_exp = phi_exp_deg

ab_dev = read(cat_hdu, "ab_dev")
re_dev = theta_dev
phi_dev = phi_dev_deg


# What do we do for ab, angle, and scale when the 
# galaxy is a mix of types?
CatalogEntry(
	[ra[row], dec[row]], # pos
	true, # is_star
    starflux[:, row], # star_fluxes
    gal_flux, # gal_fluxes
    fracdev[row], # gal_frac_dev
    # gal_ab
)
    pos::Vector{Float64}
    is_star::Bool
    star_fluxes::Vector{Float64}
    gal_fluxes::Vector{Float64}
    gal_frac_dev::Float64
    gal_ab::Float64
    gal_angle::Float64
    gal_scale::Float64
end

rows = collect(1:length(read(cat_hdu, "objid")))

# Skip "bright" rows and those with prob_psf not in [0, 1].
rows = rows[not_bright & no_child & (!is_bad)]
catalog = Array(CatalogEntry, length(rows));
i = 1

sum(is_bad & (!(not_bright & no_child)))
for row=rows

	gal_flux = zeros(length(band_letters))
	if is_comp:
		gal_flux = compflux[row, :]
	elseif has_dev:
		gal_flux = devflux[row, :]
	elseif has_exp:
		exp_flux = expflux[row, :]
	else:
		error("Should skip rows with a bad devflux.")
	end


end

# Note that I don't think we need / use the brightness
# values, so don't need flux2bright = nmgy2bright


