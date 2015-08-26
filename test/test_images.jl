using Celeste
using Base.Test
using CelesteTypes
using SampleData
using DataFrames

import Images
import SloanDigitalSkySurvey: SDSS

println("Running Images tests.")

field_dir = joinpath(dat_dir, "sample_field")
run_num = "003900"
camcol_num = "6"
field_num = "0269"
band_letters = ['u', 'g', 'r', 'i', 'z']

function test_blob()
  blob = Images.load_sdss_blob(field_dir, run_num, camcol_num, field_num);
  cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);
  cat_entries = Images.convert_catalog_to_celeste(cat_df, blob);

  # Just check some basic facts about the catalog.
  @test nrow(cat_df) == 805
  @test length(cat_entries) == nrow(cat_df)
  @test sum([ cat_entry.is_star for cat_entry in cat_entries ]) ==
        sum(cat_df[:is_star])

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
  Images.crop_image!(blob, width, obj_loc)
  for b=1:5
    @test 2 * width <= blob[b].H <= 2 * (width + 1)
    @test 2 * width <= blob[b].W <= 2 * (width + 1)
  end
  @test Images.test_catalog_entry_in_image(blob, obj_loc)
end

test_blob()
