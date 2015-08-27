using Celeste
using Base.Test
using DataFrames
import PyPlot
import Images
import SloanDigitalSkySurvey: SDSS

const field_dir =
  joinpath(Pkg.dir("SloanDigitalSkySurvey"), "dat", "sample_field")
const run_num = "003900"
const camcol_num = "6"
const field_num = "0269"

# Load the catalog and select an object.
cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);
objid = cat_df[:objid][1]

# Load the point spread function.
raw_psf_comp =
  SDSS.load_psf_data(field_dir, run_num, camcol_num, field_num, b);

blob = Images.load_sdss_blob(field_dir, run_num, camcol_num, field_num,
  mask_planes=Set());

mask_planes = ["S_MASK_INTERP", "S_MASK_SATUR", "S_MASK_CR", "S_MASK_GHOST"]
blob_masks = Dict()
for mask_plane in mask_planes
  println("getting $mask_plane")
  blob_masks[mask_plane] = deepcopy(blob)
  for b=1:5
    SDSS.mask_image!(blob[b].pixels, field_dir, run_num, camcol_num, field_num, b,
                     mask_planes=Set({mask_plane}))
  end
end

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
