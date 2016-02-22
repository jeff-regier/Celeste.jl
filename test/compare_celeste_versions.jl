# Compare the new Celeste version to an old version in another repo.

# To track memory, use:
# julia --track-allocation=user

#include("test/debug_with_master.jl");
#using Debug

# blob, mp, bodies, tiled_blob = Debug.SampleData.gen_two_body_dataset();
# @time debug_elbo = Debug.ElboDeriv.elbo(tiled_blob, mp);

using Celeste
using CelesteTypes
using Base.Test
using SampleData
import Synthetic

profile_n = 0

blob, mp, bodies, tiled_blob = gen_two_body_dataset();
mp.active_sources = [2];
@time elbo = ElboDeriv.elbo(tiled_blob, mp);
elbo.v
size(elbo.d)

mp.active_sources = [1, 2];
@time elbo = ElboDeriv.elbo(tiled_blob, mp);
elbo.v
size(elbo.d)



println(elbo.d[1, 1])
@test_approx_eq elbo.v[1] debug_elbo.v[1]
@test_approx_eq elbo.d debug_elbo.d




##########
@time ElboDeriv.elbo_likelihood(tiled_blob, mp, calculate_derivs=false);
@time ElboDeriv.elbo_likelihood(tiled_blob, mp, calculate_hessian=false);
@time ElboDeriv.elbo_likelihood(tiled_blob, mp);

Profile.clear_malloc_data()
Profile.clear()
@profile for i = 1:profile_n
  println(".")
  ElboDeriv.elbo_likelihood(tiled_blob, mp, calculate_hessian=true);
  #Debug.ElboDeriv.elbo_likelihood(tiled_blob, mp);
end
#Profile.print()

# To see memory, quit and run
using Coverage
res = analyze_malloc("src");
pn = 40; [ println(res[end - (pn - i)]) for i=1:pn ];



# Checking if the ELBO is different between the two versions.
#include("src/ElboDeriv.jl")


blob, mp, bodies, tiled_blob = Debug.SampleData.gen_two_body_dataset();
@time debug_elbo = Debug.ElboDeriv.elbo(tiled_blob, mp);


@time Debug.ElboDeriv.elbo(tiled_blob, mp);
#@time ElboDeriv.elbo_likelihood(tiled_blob, mp);
@time ElboDeriv.elbo(tiled_blob, mp, calculate_derivs=false);
@time ElboDeriv.elbo(tiled_blob, mp, calculate_hessian=false);
@time ElboDeriv.elbo(tiled_blob, mp);

Profile.clear_malloc_data()
Profile.clear()
@profile for i = 1:profile_n
  #ElboDeriv.elbo_likelihood(tiled_blob, mp, calculate_hessian=false);
  Debug.ElboDeriv.elbo_likelihood(tiled_blob, mp);
end
#Profile.print(format=:tree, sortedby = :count)

# To see memory, quit and run
using Coverage
res = analyze_malloc("src")
#res = analyze_malloc("/home/rgiordan/Documents/git_repos/debugging_branch_Celeste.jl/CelesteDebug.jl/src");
pn = 40; [ println(res[end - (pn - i)]) for i=1:pn ];





function evaluate_their_elbo()
  blob, mp, bodies, tiled_blob = Debug.SampleData.gen_two_body_dataset();
  Debug.ElboDeriv.elbo(tiled_blob, mp)
end

function evaluate_our_elbo()
  blob, mp, bodies, tiled_blob = gen_two_body_dataset();
  ElboDeriv.elbo(tiled_blob, mp)
end

their_elbo = evaluate_their_elbo();
our_elbo = evaluate_our_elbo();

@test_approx_eq our_elbo.v[1] their_elbo.v[1]



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

@test_approx_eq their_fs0m.v[1] our_fs0m.v[1]
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

@test_approx_eq their_fs1m.v[1] our_fs1m.v[1]
@test_approx_eq their_fs1m.d our_fs1m.d


# Check the brightnesses.  Looks good.

blob, mp, bodies, tiled_blob = gen_two_body_dataset();
@time their_sbs = [Debug.ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S];
@time our_sbs = ElboDeriv.load_source_brightnesses(mp);

for s=1:length(our_sbs), b=1:5, i=1:2
  @test_approx_eq our_sbs[s].E_l_a[b, i].v[1] their_sbs[s].E_l_a[b, i].v[1]
  @test_approx_eq our_sbs[s].E_l_a[b, i].d their_sbs[s].E_l_a[b, i].d
  @test_approx_eq our_sbs[s].E_ll_a[b, i].v[1] their_sbs[s].E_ll_a[b, i].v[1]
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
  tile_sources = mp.tile_sources[b][1,1]

  star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
  sbs = ElboDeriv.load_source_brightnesses(mp);
  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, mp.S, mp.S);
  @time ElboDeriv.populate_fsm_vecs!(
    elbo_vars, mp, tile_sources, tile, h, w, sbs, gal_mcs, star_mcs);
  clear!(elbo_vars.E_G);
  clear!(elbo_vars.var_G);

  ElboDeriv.accumulate_source_brightness!(elbo_vars, mp, sbs, s, b);
  deepcopy(elbo_vars.E_G), deepcopy(elbo_vars.var_G)
end



#@time their_E_G, their_var_G = their_accumulate_pixel_stats();
@time our_E_G, our_var_G = our_accumulate_pixel_stats();

@test_approx_eq their_E_G.v[1] our_E_G.v[1]
@test_approx_eq their_E_G.d our_E_G.d

@test_approx_eq their_var_G.v[1] our_E_G2.v[1] - (our_E_G.v[1] ^ 2)

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
  tile_sources = mp.tile_sources[b][1,1];

  star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
  sbs = ElboDeriv.load_source_brightnesses(mp);
  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, mp.S, mp.S);
  ElboDeriv.populate_fsm_vecs!(
    elbo_vars, mp, tile_sources, tile, h, w, sbs, gal_mcs, star_mcs);

  this_pixel = tile.pixels[h, w]
  ElboDeriv.get_expected_pixel_brightness!(
    elbo_vars, h, w, sbs, star_mcs, gal_mcs, tile,
    mp, tile_sources, include_epsilon=true)

  deepcopy(elbo_vars.E_G), deepcopy(elbo_vars.var_G)
end

#their_E_G, their_var_G = their_expected_pixel_brightness();
@time our_E_G, our_var_G = our_expected_pixel_brightness();

@test_approx_eq their_E_G.v[1] our_E_G.v[1]
@test_approx_eq their_var_G.v[1] our_var_G.v[1]





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
  tile_sources = mp.tile_sources[b][1,1];

  star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
  sbs = ElboDeriv.load_source_brightnesses(mp);
  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, mp.S, mp.S);
  ElboDeriv.populate_fsm_vecs!(
    elbo_vars, mp, tile_sources, tile, h, w, sbs, gal_mcs, star_mcs);
  clear!(elbo_vars.E_G);
  clear!(elbo_vars.var_G);

  this_pixel = tile.pixels[h, w]
  ElboDeriv.get_expected_pixel_brightness!(
    elbo_vars, h, w, sbs, star_mcs, gal_mcs, tile,
    mp, tile_sources, include_epsilon=true)


  println(elbo_vars.elbo.v[1])
  iota = tile.constant_background ? tile.iota : tile.iota_vec[h]
  ElboDeriv.add_elbo_log_term!(elbo_vars, this_pixel, iota)
  println("Log term: ", elbo_vars.elbo.v[1])
  CelesteTypes.add_scaled_sfs!(elbo_vars.elbo, elbo_vars.E_G, scale=-iota)
  println(elbo_vars.elbo.v[1])

  deepcopy(elbo_vars.elbo)
end

#their_accum = their_accum_pixel_ret();
@time our_accum = our_accum_pixel_ret();

@test_approx_eq their_accum.v[1] our_accum.v[1]


############################
############################
# Check each step


using Celeste
using CelesteTypes
using Base.Test
using SampleData
import Synthetic

#include("src/ElboDeriv.jl")

s = 1
b = 1
h = 10
w = 10
blob, mp, bodies, tiled_blob = gen_two_body_dataset();
tile = tiled_blob[b][1,1];
tile_sources = mp.tile_sources[b][1,1];

#@time elbo = ElboDeriv.elbo(tiled_blob, mp);

@time star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
@time sbs = ElboDeriv.load_source_brightnesses(mp);
@time elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, mp.S, mp.S);
@time ElboDeriv.populate_fsm_vecs!(
  elbo_vars, mp, tile_sources, tile, h, w, sbs, gal_mcs, star_mcs);
clear!(elbo_vars.E_G);
clear!(elbo_vars.var_G);

this_pixel = tile.pixels[h, w]

@time ElboDeriv.get_expected_pixel_brightness!(
  elbo_vars, h, w, sbs, star_mcs, gal_mcs, tile,
  mp, tile_sources, include_epsilon=true);



Profile.clear_malloc_data()
Profile.clear()
profile_n = 50
@profile for i = 1:profile_n
  print(".")
  ElboDeriv.get_expected_pixel_brightness!(
    elbo_vars, h, w, sbs, star_mcs, gal_mcs, tile,
    mp, tile_sources, include_epsilon=true);
end
println("Done.")


@time ElboDeriv.populate_fsm_vecs!(
  elbo_vars, mp, tile_sources, tile, h, w, sbs, gal_mcs, star_mcs)

# # This combines the sources into a single brightness value for the pixel.
@time ElboDeriv.combine_pixel_sources!(elbo_vars, mp, tile_sources, tile, sbs);

# Dig into combine_pixel_sources/.  We expect 31k of allocations.
clear!(elbo_vars.E_G,
  clear_hessian=elbo_vars.calculate_hessian && elbo_vars.calculate_derivs);
clear!(elbo_vars.var_G,
  clear_hessian=elbo_vars.calculate_hessian && elbo_vars.calculate_derivs);

# This would get run twice.
s = tile_sources[1]
@time active_source = s in mp.active_sources
@time calculate_hessian =
  elbo_vars.calculate_hessian && elbo_vars.calculate_derivs &&
  active_source
# It's all in here:
@time ElboDeriv.accumulate_source_brightness!(elbo_vars, mp, sbs, s, tile.b)
@time sa = findfirst(mp.active_sources, s)[1]
@time ElboDeriv.add_sources_sf!(elbo_vars.E_G, elbo_vars.E_G_s, sa,
  calculate_hessian=calculate_hessian);
@time ElboDeriv.add_sources_sf!(elbo_vars.var_G, elbo_vars.var_G_s, sa,
  calculate_hessian=calculate_hessian);
