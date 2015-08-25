using Celeste
using Base.Test
using CelesteTypes
using SampleData

import SDSS

println("Running sdss tests.")

const field_dir = joinpath(dat_dir, "sample_field")
const run_num = "003900"
const camcol_num = "6"
const field_num = "0269"
const band_letters = ['u', 'g', 'r', 'i', 'z']

function test_load_catalog()
  cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);
  cat_entries = SDSS.convert_catalog_to_celeste(cat_df, blob);

  # Just check some basic facts about the catalog.
  @test nrow(cat_df) == 805
  @test length(cat_entries) == nrow(cat_df)
  @test sum([ cat_entry.is_star for cat_entry in cat_entries ]) ==
        sum(cat_df[:is_star])
end

function test_load_blob()
  b_letter = band_letters[1]
  blob = SDSS.load_sdss_blob(field_dir, run_num, camcol_num, field_num);
end

function test_load_psf()
  raw_psf_comp =
    SDSS.load_psf_data(field_dir, run_num, camcol_num, field_num, 1);
  raw_psf = PSF.get_psf_at_point(psf_point_x, psf_point_y, raw_psf_comp);

  # Check that the raw psf is approximately symmetric
  center_point = (size(raw_psf)[1] + 1) / 2
  @test_approx_eq_eps(sum(collect(1:size(raw_psf)[1]) .* raw_psf) / sum(raw_psf),
                      center_point, 0.5)
  @test_approx_eq_eps(sum(collect(1:size(raw_psf)[1])' .* raw_psf) / sum(raw_psf),
                      center_point, 0.5)

end


function test_field()
  b = 1
  band_gain, band_dark_variance =
    SDSS.load_photo_field(field_dir, run_num, camcol_num, field_num)
  nelec, calib_col, sky_grid, sky_x, sky_y, sky_image, wcs =
    SDSS.load_raw_field(field_dir, run_num, camcol_num, field_num,
                        b, band_gain[b])

  @test size(sky_image) == size(nelec)
  @test length(calib_col) == size(nelec)[1]
  @test sky_image[10, 5] == sky_grid[sky_x[10], sky_y[5]]

  nelec_original = deepcopy(nelec)
  SDSS.mask_image!(nelec, field_dir, run_num, camcol_num, field_num, b);

  # Test that at least some pixels are bad
  @test sum(isnan(nelec)) > 0

  # Test that no more than 1% are bad.  Not sure this is a good test
  # in general but it works for the current files.
  @test sum(isnan(nelec)) / prod(size(nelec)) < 0.01
end


function test_blob()
  blob = SDSS.load_sdss_blob(field_dir, run_num, camcol_num, field_num);
  cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);

  # Find an object near the middle of the image.
  img_center = Float64[ median(cat_df[:ra]), median(cat_df[:dec]) ]
  dist = by(cat_df, :objid,
     df -> DataFrame(dist=(df[:ra] - img_center[1]).^2 +
                          (df[:dec] - img_center[2]).^2))
  obj_rows = dist[:dist] .== minimum(dist[:dist])
  @assert any(obj_rows)
  obj_loc = Float64[ cat_df[obj_rows, :ra][1], cat_df[obj_rows, :dec][1]]
  objid = cat_df[obj_rows, :objid][1]

  # Test cropping.
  original_blob = deepcopy(blob)
  width = 5.0
  SDSS.crop_image!(blob, width, obj_loc)
  for b=1:5
    @test 2 * width <= blob[b].H <= 2 * (width + 1)
    @test 2 * width <= blob[b].W <= 2 * (width + 1)
  end
  @test SDSS.test_catalog_entry_in_image(blob, obj_loc)
end

test_load_catalog()
test_load_psf()
test_field()
test_blob()
