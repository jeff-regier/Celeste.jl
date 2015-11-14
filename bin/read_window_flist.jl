# That file for DR9 is here:
# http://data.sdss3.org/sas/dr9/env/PHOTO_RESOLVE/window_flist.fits
#
# Its description is here:
# http://data.sdss3.org/datamodel/files/PHOTO_RESOLVE/window_flist.html
#
# Either cut on incl < 1 to get stripe 82, or cut on |Dec| < 1.5.
#
# You can use
# http://skyserver.sdss.org/dr12/en/tools/chart/navi.aspx?ra=75&dec=0
# ...to compare with downloded images.

using Celeste
using CelesteTypes
using DataFrames
using StatsBase

import FITSIO
import PyPlot
import SampleData.dat_dir


n, bins, patches = PyPlot.hist(x, 50, normed=1, facecolor="g", alpha=0.75)
PyPlot.show()

window_flist = FITSIO.FITS(joinpath(dat_dir, "window_flist.fits"));

num_cols = FITSIO.read_key(window_flist[2], "TFIELDS")[1]
ttypes = [FITSIO.read_key(window_flist[2], "TTYPE$i")[1] for i in 1:num_cols];

keep_fields =
  ["RUN", "RERUN", "CAMCOL", "FIELD", "MJD", "RA", "DEC", "INCL",
   "PHOTO_STATUS", "PSP_STATUS", "IMAGE_STATUS", "CALIB_STATUS", "SCORE"]

df = DataFrames.DataFrame()
for i in findin(ttypes, keep_fields)
    println(ttypes[i])
    tmp_data = read(window_flist[2], ttypes[i]);

    if length(size(tmp_data)) > 1
      for row in 1:(size(tmp_data)[1])
        df[symbol(string(ttypes[i], row))] = tmp_data[row,:][:]
      end
    else
      df[symbol(ttypes[i])] = tmp_data
    end
end

FITSIO.close(window_flist)

is_stripe_82 = (abs(df[:DEC]) .< 1.25) & (4 * 15 .< df[:RA] .< 21 * 15);
good_stripe_82 = is_stripe_82 & (df[:SCORE] .> 0.6);
sum(df[:INCL] .< 1)
sum(df[:DEC] .< 1.5)
sum(abs(df[:DEC]) .< 1.25)
sum(is_stripe_82)
sum(good_stripe_82)

df[good_stripe_82, [:RA, :DEC]]

# Location as RA, DEC
loc = [74.9771, -0.06155];
dist_to_loc = sqrt((df[:RA] - loc[1]) .^ 2 + (df[:DEC] - loc[2]) .^ 2);
df[:DIST] = dist_to_loc;
top_n = sortperm(dist_to_loc)[1:10];
df[top_n, [:RUN, :RERUN, :CAMCOL, :FIELD, :DIST]]

function download_command(df_row::Int64)
  run = df[df_row, :RUN]
  camcol = df[df_row, :CAMCOL]
  field = df[df_row, :FIELD]
  string("./download_fits_files.py ",
         "--run=$run --camcol=$camcol --field=$field ",
         "--destination_dir=$(dat_dir)")
end

############################
# Status indicators

# The PSP_STATUS is the PSF status:
# http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/psField.html
countmap(df[:PSP_STATUS1])

# The IMAGE_STATUS is this bitmap:
# http://www.sdss3.org/dr10/algorithms/bitmask_image_status.php
# Why is it never 0?
countmap(df[:IMAGE_STATUS1])

# The CALIB_STATUS is here:
# http://www.sdss3.org/dr10/algorithms/bitmask_calib_status.php
# It is always zero.
countmap(df[:CALIB_STATUS1])

# PHOTO_STATUS is supposedly (but not actually) defined here:
# http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/fpFieldStat.html
# It is only 0 or 3.
countmap(df[:PHOTO_STATUS])

# The SCORE field is defined here:
# http://www.sdss3.org/dr10/algorithms/resolve.php
# > 0.6 means photometric.
sum(df[:SCORE] .> 0.6) / size(df)[1]

# PyPlot.plt[:hist](df[:MJD])
