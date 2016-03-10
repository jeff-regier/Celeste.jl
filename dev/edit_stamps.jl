#!/usr/bin/env julia

using Celeste: Types, SampleData, Transform
import Celeste: ModelInit, SkyImages, ElboDeriv, Synthetic
import Celeste.Synthetic
using Distributions
using DataFrames
import SloanDigitalSkySurvey: SDSS, WCSUtils, PSF

using PyPlot

band_letter = "i"

# Ensure that test images are available.
stamp_dir = joinpath(Pkg.dir("Celeste"), "test", "data")
stamp_id = "164.4311-39.0359"
filename = "$stamp_dir/stamp-$band_letter-$stamp_id.fits"
filename_out = "$stamp_dir/stamp-$band_letter-$(stamp_id)_2kpsf.fits"

fits = FITSIO.FITS(filename, "r")
hdr = FITSIO.read_header(fits[1]);
original_pixels = read(fits[1]);
close(fits)

# Read the PSF, which we will modify.
alphaBar = [hdr["PSF_P0"]; hdr["PSF_P1"]; hdr["PSF_P2"]]
xiBar = [
    [hdr["PSF_P3"]  hdr["PSF_P4"]];
    [hdr["PSF_P5"]  hdr["PSF_P6"]];
    [hdr["PSF_P7"]  hdr["PSF_P8"]]]'

tauBar = Array(Float64, 2, 2, 3)
tauBar[:,:,1] = [[hdr["PSF_P9"] hdr["PSF_P11"]];
                 [hdr["PSF_P11"] hdr["PSF_P10"]]]
tauBar[:,:,2] = [[hdr["PSF_P12"] hdr["PSF_P14"]];
                 [hdr["PSF_P14"] hdr["PSF_P13"]]]
tauBar[:,:,3] = [[hdr["PSF_P15"] hdr["PSF_P17"]];
                 [hdr["PSF_P17"] hdr["PSF_P16"]]]

psf_original = [PsfComponent(alphaBar[k], xiBar[:, k],
                    tauBar[:, :, k]) for k in 1:3]
rows = cols = collect(linspace(-10, 10, 50));
scale = minimum(diff(rows))

psf_original_rendered =
  SkyImages.get_psf_at_point(psf_original, rows=rows, cols=cols);

optim_result, mu_vec, sigma_vec, weight_vec =
  PSF.fit_psf_gaussians_least_squares(psf_original_rendered, K=2, verbose=true);

# Return to the original scale.  Note that since it is a density, the
# weights need to change as well.
psf_new = [PsfComponent(weight_vec[k] * (scale ^ 2),
                        mu_vec[k] .* scale,
                        sigma_vec[k] .* (scale ^ 2)) for k in 1:2]

psf_new_rendered =
  SkyImages.get_psf_at_point(psf_new, rows=rows, cols=cols);

matshow(psf_original_rendered); colorbar()
matshow(psf_new_rendered); colorbar()
matshow(psf_new_rendered - psf_original_rendered); colorbar()

@assert sqrt(mean((psf_original_rendered - psf_new_rendered) .^ 2)) < 0.005


# # This is not necessary.
# new_header = FITSIO.FITSHeader(ASCIIString[], ASCIIString[], ASCIIString[])
# for (key in keys(hdr))
#   if ismatch(r"^PSF_P[0-9]+", key)
#     println(key)
#   else
#     println("non-match $key")
#     new_header[key] = hdr[key]
#     FITSIO.set_comment!(new_header, key, FITSIO.get_comment(hdr, key))
#   end
# end

new_header = deepcopy(hdr);

# The weights
for k=1:3
  weight_ind = k - 1
  mean_ind = 3 + (k - 1) * 2
  sigma_ind = 9 + (k - 1) * 3
  if k < 3
    new_header["PSF_P$(weight_ind)"] = psf_new[k].alphaBar

    new_header["PSF_P$(mean_ind)"] = psf_new[k].xiBar[1]
    new_header["PSF_P$(mean_ind + 1)"] = psf_new[k].xiBar[2]

    # The order is (xx), (yy), (xy)
    new_header["PSF_P$(sigma_ind)"] = psf_new[k].tauBar[1, 1]
    new_header["PSF_P$(sigma_ind + 1)"] = psf_new[k].tauBar[2, 2]
    new_header["PSF_P$(sigma_ind + 2)"] = psf_new[k].tauBar[1, 2]
  else
    # Set the third component to zero to avoid changing the stamp file format..
    new_header["PSF_P$(weight_ind)"] = 0.0

    new_header["PSF_P$(mean_ind)"] = 0.0
    new_header["PSF_P$(mean_ind + 1)"] = 0.0

    # The order is (xx), (yy), (xy)
    new_header["PSF_P$(sigma_ind)"] = 1.0
    new_header["PSF_P$(sigma_ind + 1)"] = 1.0
    new_header["PSF_P$(sigma_ind + 2)"] = 1.0
  end
end

# Write to a new file
fits_out = FITSIO.FITS(filename_out, "w")
FITSIO.write(fits_out, original_pixels, header=new_header);
close(fits_out)
