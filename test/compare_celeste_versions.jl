# Compare the new Celeste version to an old version in another repo.

# To track memory, use:
# julia --track-allocation=user

using Celeste
using CelesteTypes
using Base.Test
using SampleData
import Synthetic

blob, mp, bodies, tiled_blob = gen_two_body_dataset();
@time ElboDeriv.elbo_likelihood(tiled_blob, mp, calculate_hessian=false);

Profile.clear_malloc_data()
Profile.clear()
@time ElboDeriv.elbo_likelihood(tiled_blob, mp, calculate_hessian=false);

# To see memory, quit and run
using Coverage
res = analyze_malloc("src");
res[(end - 20):end]



@time ElboDeriv.elbo_likelihood(tiled_blob, mp, calculate_derivs=true);
@time ElboDeriv.elbo_likelihood(tiled_blob, mp, calculate_hessian=false);
@time ElboDeriv.elbo_likelihood(tiled_blob, mp, calculate_derivs=false);





# Checking if the ELBO is different between the two versions.
include("test/debug_with_master.jl"); include("src/ElboDeriv.jl")

using Debug

function evaluate_their_elbo()
  blob, mp, bodies, tiled_blob = Debug.SampleData.gen_two_body_dataset();
  Debug.ElboDeriv.elbo_likelihood(tiled_blob, mp)
end

function evaluate_our_elbo()
  blob, mp, bodies, tiled_blob = gen_two_body_dataset();
  ElboDeriv.elbo_likelihood(tiled_blob, mp)
end

their_elbo = evaluate_their_elbo();
our_elbo = evaluate_our_elbo();

@test_approx_eq our_elbo.v their_elbo.v

blob, mp, bodies, tiled_blob = gen_two_body_dataset();
@time Debug.ElboDeriv.elbo_likelihood(tiled_blob, mp);
@time ElboDeriv.elbo_likelihood(tiled_blob, mp);
@time ElboDeriv.elbo_likelihood(tiled_blob, mp, calculate_hessian=false);
@time ElboDeriv.elbo_likelihood(tiled_blob, mp, calculate_derivs=false);

Profile.clear_malloc_data()
Profile.clear()
@profile ElboDeriv.elbo_likelihood(tiled_blob, mp, calculate_hessian=false);
#@profile Debug.ElboDeriv.elbo_likelihood(tiled_blob, mp);
Profile.print(format=:flat)





# Deeper checks:
# fsXm looks good.
b = 1
x = [10., 9.]
wcs_jacobian = eye(Float64, 2);

function get_their_fs0m()
  blob, mp, bodies, tiled_blob = Debug.SampleData.gen_two_body_dataset();

  star_mcs, gal_mcs = Debug.ElboDeriv.load_bvn_mixtures(mp, b);
  fs0m = zero_sensitive_float(StarPosParams, Float64);
  Debug.ElboDeriv.accum_star_pos!(star_mcs[1], x, fs0m, wcs_jacobian);
  fs0m
end

function get_our_fs0m()
  blob, mp, bodies, tiled_blob = gen_two_body_dataset();

  s = 1
  star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, 1, 1);
  ElboDeriv.accum_star_pos!(elbo_vars, s, star_mcs[1], x, wcs_jacobian);
  elbo_vars.fs0m_vec[s]
end

@time their_fs0m = get_their_fs0m();
@time our_fs0m = get_our_fs0m();

@test_approx_eq their_fs0m.v our_fs0m.v
@test_approx_eq their_fs0m.d our_fs0m.d


function get_their_fs1m()
  blob, mp, bodies, tiled_blob = Debug.SampleData.gen_two_body_dataset();

  star_mcs, gal_mcs = Debug.ElboDeriv.load_bvn_mixtures(mp, b);
  fs1m = zero_sensitive_float(GalaxyPosParams, Float64);
  Debug.ElboDeriv.accum_galaxy_pos!(gal_mcs[1], x, fs1m, wcs_jacobian);
  fs1m
end

function get_our_fs1m()
  blob, mp, bodies, tiled_blob = gen_two_body_dataset();

  s = 1
  star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, 1, 1);
  ElboDeriv.accum_galaxy_pos!(elbo_vars, s, gal_mcs[1], x, wcs_jacobian);
  elbo_vars.fs1m_vec[s]
end

@time their_fs1m = get_their_fs1m();
@time our_fs1m = get_our_fs1m();

@test_approx_eq their_fs1m.v our_fs1m.v
@test_approx_eq their_fs1m.d our_fs1m.d


# Check the brightnesses.  Looks good.

blob, mp, bodies, tiled_blob = gen_two_body_dataset();
@time their_sbs = [Debug.ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S];
@time our_sbs = ElboDeriv.load_source_brightnesses(mp);

for s=1:length(our_sbs), b=1:5, i=1:2
  @test_approx_eq our_sbs[s].E_l_a[b, i].v their_sbs[s].E_l_a[b, i].v
  @test_approx_eq our_sbs[s].E_l_a[b, i].d their_sbs[s].E_l_a[b, i].d
  @test_approx_eq our_sbs[s].E_ll_a[b, i].v their_sbs[s].E_ll_a[b, i].v
  @test_approx_eq our_sbs[s].E_ll_a[b, i].d their_sbs[s].E_ll_a[b, i].d
end


##################################
# Check a pixel accumlation.
s = 2
b = 1
h = 10
w = 10
wcs_jacobian = Float64[1 0; 0 1]

function their_accumulate_pixel_stats()
  blob, mp, bodies, tiled_blob = Debug.SampleData.gen_two_body_dataset();
  tile = tiled_blob[b][1,1];

  #zsf = Debug.CelesteTypes.zero_sensitive_float;
  zsf = zero_sensitive_float;

  star_mcs, gal_mcs = Debug.ElboDeriv.load_bvn_mixtures(mp, b);
  # I don't understand this:
  # fs0m = zsf(Debug.CelesteTypes.StarPosParams, Float64);
  # fs1m = zsf(Debug.CelesteTypes.GalaxyPosParams, Float64);
  # E_G = zsf(Debug.CelesteTypes.CanonicalParams, Float64, mp.S);
  # var_G = zsf(Debug.CelesteTypes.CanonicalParams, Float64, mp.S);
  fs0m = zsf(CelesteTypes.StarPosParams, Float64);
  fs1m = zsf(CelesteTypes.GalaxyPosParams, Float64);
  E_G = zsf(CelesteTypes.CanonicalParams, Float64, mp.S);
  var_G = zsf(CelesteTypes.CanonicalParams, Float64, mp.S);

  sbs = [Debug.ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S];

  m_pos = Float64[tile.h_range[h], tile.w_range[w]]
  Debug.ElboDeriv.accum_pixel_source_stats!(
      sbs[s], star_mcs, gal_mcs,
      mp.vp[s], s, s, m_pos, b, fs0m, fs1m, E_G, var_G, wcs_jacobian)

  deepcopy(E_G), deepcopy(var_G)
end


function our_accumulate_pixel_stats()
  blob, mp, bodies, tiled_blob = gen_two_body_dataset();
  tile = tiled_blob[b][1,1];

  star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
  sbs = ElboDeriv.load_source_brightnesses(mp);
  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, mp.S, mp.S);
  ElboDeriv.populate_fsm_vecs!(
    elbo_vars, mp, tile, h, w, sbs, gal_mcs, star_mcs);
  clear!(elbo_vars.E_G);
  clear!(elbo_vars.E_G2);

  ElboDeriv.accumulate_source_brightness!(elbo_vars, mp, sbs, s, b);
  deepcopy(elbo_vars.E_G), deepcopy(elbo_vars.E_G2)
end



their_E_G, their_var_G = their_accumulate_pixel_stats();
our_E_G, our_E_G2 = our_accumulate_pixel_stats();

@test_approx_eq their_E_G.v our_E_G.v
@test_approx_eq their_E_G.d our_E_G.d

@test_approx_eq their_var_G.v our_E_G2.v - (our_E_G.v ^ 2)

################################
# It might be in the accumulation of E_G2 across multiple sources.

s = 1
b = 1
h = 10
w = 10
wcs_jacobian = Float64[1 0; 0 1]


function their_expected_pixel_brightness()
  blob, mp, bodies, tiled_blob = Debug.SampleData.gen_two_body_dataset();
  tile = tiled_blob[b][1,1];

  zsf = zero_sensitive_float;

  star_mcs, gal_mcs = Debug.ElboDeriv.load_bvn_mixtures(mp, b);

  accum = zsf(CelesteTypes.CanonicalParams, Float64, mp.S);
  fs0m = zsf(CelesteTypes.StarPosParams, Float64);
  fs1m = zsf(CelesteTypes.GalaxyPosParams, Float64);
  E_G = zsf(CelesteTypes.CanonicalParams, Float64, mp.S);
  var_G = zsf(CelesteTypes.CanonicalParams, Float64, mp.S);

  sbs = Debug.ElboDeriv.SourceBrightness{Float64}[
    Debug.ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S];

  m_pos = Float64[tile.h_range[h], tile.w_range[w]]

  this_pixel = tile.pixels[h, w]
  iota = Debug.ElboDeriv.expected_pixel_brightness!(
    h, w, sbs, star_mcs, gal_mcs, tile, E_G, var_G,
    mp, mp.active_sources, fs0m, fs1m,
    include_epsilon=true);
  deepcopy(E_G), deepcopy(var_G)
end



function our_expected_pixel_brightness()
  blob, mp, bodies, tiled_blob = gen_two_body_dataset();
  tile = tiled_blob[b][1,1];

  star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
  sbs = ElboDeriv.load_source_brightnesses(mp);
  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, mp.S, mp.S);
  ElboDeriv.populate_fsm_vecs!(
    elbo_vars, mp, tile, h, w, sbs, gal_mcs, star_mcs);

  this_pixel = tile.pixels[h, w]
  ElboDeriv.get_expected_pixel_brightness!(
    elbo_vars, h, w, sbs, star_mcs, gal_mcs, tile,
    mp, include_epsilon=true)

  deepcopy(elbo_vars.E_G), deepcopy(elbo_vars.var_G)
end

their_E_G, their_var_G = their_expected_pixel_brightness();
our_E_G, our_var_G = our_expected_pixel_brightness();

@test_approx_eq their_E_G.v our_E_G.v
@test_approx_eq their_var_G.v our_var_G.v





##################################
# test accum_pixel_ret!

s = 1
b = 1
h = 10
w = 10
wcs_jacobian = Float64[1 0; 0 1]


function their_accum_pixel_ret()
  blob, mp, bodies, tiled_blob = Debug.SampleData.gen_two_body_dataset();
  tile = tiled_blob[b][1,1];

  zsf = zero_sensitive_float;

  star_mcs, gal_mcs = Debug.ElboDeriv.load_bvn_mixtures(mp, b);

  accum = zsf(CelesteTypes.CanonicalParams, Float64, mp.S);
  fs0m = zsf(CelesteTypes.StarPosParams, Float64);
  fs1m = zsf(CelesteTypes.GalaxyPosParams, Float64);
  E_G = zsf(CelesteTypes.CanonicalParams, Float64, mp.S);
  var_G = zsf(CelesteTypes.CanonicalParams, Float64, mp.S);

  sbs = Debug.ElboDeriv.SourceBrightness{Float64}[
    Debug.ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S];

  m_pos = Float64[tile.h_range[h], tile.w_range[w]]

  this_pixel = tile.pixels[h, w]
  iota = Debug.ElboDeriv.expected_pixel_brightness!(
    h, w, sbs, star_mcs, gal_mcs, tile, E_G, var_G,
    mp, mp.active_sources, fs0m, fs1m,
    include_epsilon=true);

  Debug.ElboDeriv.accum_pixel_ret!(
    mp.active_sources, this_pixel, iota, E_G, var_G, accum);
  deepcopy(accum)
end



function our_accum_pixel_ret()
  blob, mp, bodies, tiled_blob = gen_two_body_dataset();
  tile = tiled_blob[b][1,1];

  star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
  sbs = ElboDeriv.load_source_brightnesses(mp);
  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, mp.S, mp.S);
  ElboDeriv.populate_fsm_vecs!(
    elbo_vars, mp, tile, h, w, sbs, gal_mcs, star_mcs);
  clear!(elbo_vars.E_G);
  clear!(elbo_vars.var_G);

  this_pixel = tile.pixels[h, w]
  ElboDeriv.get_expected_pixel_brightness!(
    elbo_vars, h, w, sbs, star_mcs, gal_mcs, tile,
    mp, include_epsilon=true)


  println(elbo_vars.elbo.v)
  iota = tile.constant_background ? tile.iota : tile.iota_vec[h]
  ElboDeriv.add_elbo_log_term!(elbo_vars, this_pixel, iota)
  println("Log term: ", elbo_vars.elbo.v)
  CelesteTypes.add_scaled_sfs!(elbo_vars.elbo, elbo_vars.E_G, scale=-iota)
  println(elbo_vars.elbo.v)

  deepcopy(elbo_vars.elbo)
end

their_accum = their_accum_pixel_ret();
our_accum = our_accum_pixel_ret();

@test_approx_eq their_accum.v our_accum.v
