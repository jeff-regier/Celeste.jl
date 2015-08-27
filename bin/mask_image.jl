using SloanDigitalSkySurvey
using Base.Test
using DataFrames

import SDSS

# PyPlot is only a dependency for this example, so it's not in the REQUIRE file.
using PyPlot

const field_dir =
  joinpath(Pkg.dir("SloanDigitalSkySurvey"), "dat", "sample_field")
const run_num = "003900"
const camcol_num = "6"
const field_num = "0269"

# Load the catalog and select an object.
cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);
objid = cat_df[:objid][1]

# Gain and dark variance for all the bands are in a single file.
band_gain, band_dark_variance =
  SDSS.load_photo_field(field_dir, run_num, camcol_num, field_num);

# Load a band's image.  b can be in 1:5.
b = 3
nelec, calib_col, sky_grid, sky_x, sky_y, sky_image, wcs =
  SDSS.load_raw_field(field_dir, run_num, camcol_num, field_num,
                      b, band_gain[b]);

# Mask the image.
nelec_original = deepcopy(nelec);
SDSS.mask_image!(nelec, field_dir, run_num, camcol_num, field_num, b);

nelec_mask = Dict()
for mask in ["S_MASK_INTERP", "S_MASK_SATUR", "S_MASK_CR", "S_MASK_GHOST"]
  nelec_mask[mask] = deepcopy(nelec_original);
  SDSS.mask_image!(nelec_mask[mask], field_dir, run_num, camcol_num, field_num, b, mask_planes=Set({mask}));
end




# Load the point spread function.
raw_psf_comp =
  SDSS.load_psf_data(field_dir, run_num, camcol_num, field_num, b);


# Display the image with a dot at the object location.
pixel_graph = deepcopy(nelec_original)
pixel_graph_masked = deepcopy(nelec)
clip = 8000 # To see dimmer objects, clip the display at 8000 electrons
pixel_graph[pixel_graph .>= clip] = clip
pixel_graph_masked[pixel_graph_masked .>= clip] = clip

# Get the object location in pixel coordinates.
obj_loc  = Float64[cat_df[cat_df[:objid] .== objid, :ra][1],
                   cat_df[cat_df[:objid] .== objid, :dec][1]]
obj_px = WCS.world_to_pixel(wcs, obj_loc)

PyPlot.figure()
PyPlot.plt.subplot(1, 2, 1)
PyPlot.title("Band $b image without masking\nObj $objid")
PyPlot.imshow(pixel_graph', cmap=PyPlot.ColorMap("gray"), interpolation = "nearest")
# PyPlot uses zero indexing when plotting points.
PyPlot.scatter(obj_px[1] - 1, obj_px[2] - 1, marker="o", c="r", s=25)

PyPlot.plt.subplot(1, 2, 2)
PyPlot.title("Band $b image without masking\nObj $objid")
PyPlot.imshow(pixel_graph_masked', cmap=PyPlot.ColorMap("gray"), interpolation = "nearest")
PyPlot.scatter(obj_px[1] - 1, obj_px[2] - 1, marker="o", c="r", s=25)

# Display the point spread function at that object.
raw_psf = PSF.get_psf_at_point(obj_px[1], obj_px[2], raw_psf_comp);
PyPlot.figure()
PyPlot.title("Band $b PSF at pixel $(obj_px)\nObj $objid")
PyPlot.imshow(raw_psf', cmap=PyPlot.ColorMap("gray"), interpolation = "nearest")
