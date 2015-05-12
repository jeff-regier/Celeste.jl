using Celeste
using CelesteTypes
using Gadfly

using FITSIO
using WCSLIB
using DataFrames
using SampleData

using FITSIO

using DataFrames
using Grid


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
read_fields = ["run", "rerun", "camcol", "skyversion", "field", "nStars", "gain", "darkVariance"]
df = DataFrame()
for field in read_fields
    df[DataFrames.identifier(field)] = read(photofield_fits[2], field)
end

df[df[:field] .== int(frame_num), :]



# Read the image data.
# Documented here:
# http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html

b = 3
b_letter = ['u', 'g', 'r', 'i', 'z'][b]
img_filename = "$field_dir/frame-$b_letter-$run_num-$camcol_num-$frame_num.fits"
img_fits = FITS(img_filename)
length(img_fits) # Should be 4

# This is the sky-subtracted and calibrated image.  There are no fields in the first header.
processed_image = read(img_fits[1]);

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

sky_grid_vals = (1:1.:size(sky_image_raw)[1], 1:1.:size(sky_image_raw)[2]);
sky_grid = CoordInterpGrid(sky_grid_vals, sky_image_raw[:,:,1], BCnearest, InterpLinear);
sky_image = [ sky_grid[x, y] for x in sky_x, y in sky_y ];


# This is the calibration vector:
calib_image = read(img_fits[2])




# Old:
this_field = SDSS.load_field(field_dir, run_num, camcol_num, frame_num);
# Doesn't work:
this_cat = SDSS.load_catalog(field_dir, run_num, camcol_num, frame_num);
stamp_id = "164.4311-39.0359"
blob = SDSS.load_stamp_blob(dat_dir, stamp_id);
cat = SDSS.load_stamp_catalog(dat_dir, stamp_id, blob);