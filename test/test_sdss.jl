println("Running sdss tests.")


using Base.Test

const run_num = "003900"
const camcol_num = "6"
const field_num = "0269"


function test_psf_transforms()

  mu_vec = Vector{Float64}[ Float64[1, 2], Float64[-1, -2], Float64[1, -1] ]
  sigma_vec = Array(Matrix{Float64}, 3)
  sigma_vec[1] = Float64[ 1 0.1; 0.1 1]
  sigma_vec[2] = Float64[ 1 0.3; 0.3 2]
  sigma_vec[3] = Float64[ 0.5 0.2; 0.2 0.5]
  weight_vec = Float64[0.4, 0.6, 0.1]

  par = wrap_parameters(mu_vec, sigma_vec, weight_vec)
  mu_vec_test, sigma_vec_test, weight_vec_test = unwrap_parameters(par)

  using Base.Test

  for k=1:3
    @test_approx_eq mu_vec_test[k] mu_vec[k]
    @test_approx_eq sigma_vec_test[k] sigma_vec[k]
    @test_approx_eq weight_vec_test[k] weight_vec[k]
  end

end


function test_load_field()
  b = 1
  band_gain, band_dark_variance =
    SDSS.load_photo_field(datadir, run_num, camcol_num, field_num)
  nelec, calib_col, sky_grid, sky_x, sky_y, sky_image, wcs =
    SDSS.load_raw_field(datadir, run_num, camcol_num, field_num,
                        b, band_gain[b])

  @test size(sky_image) == size(nelec)
  @test length(calib_col) == size(nelec)[1]
  @test sky_image[10, 5] == sky_grid[sky_x[10], sky_y[5]]

  nelec_original = deepcopy(nelec)
  SDSS.mask_image!(nelec, datadir, run_num, camcol_num, field_num, b);

  # Test that at least some pixels are bad
  @test sum(isnan(nelec)) > 0

  # Test that no more than 1% are bad.  Not sure this is a good test
  # in general but it works for the current files.
  @test sum(isnan(nelec)) / prod(size(nelec)) < 0.01
end


function test_load_catalog()
  # Just check that the catalog loads.
  cat_df = SDSS.load_catalog_df(datadir, run_num, camcol_num, field_num);
  @test size(cat_df, 1) == 805
end


test_load_field()
test_load_catalog()
