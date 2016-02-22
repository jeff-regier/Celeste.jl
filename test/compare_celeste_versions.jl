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
@test_approx_eq elbo.v debug_elbo.v
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

@test_approx_eq our_elbo.v their_elbo.v



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


  println(elbo_vars.elbo.v)
  iota = tile.constant_background ? tile.iota : tile.iota_vec[h]
  ElboDeriv.add_elbo_log_term!(elbo_vars, this_pixel, iota)
  println("Log term: ", elbo_vars.elbo.v)
  CelesteTypes.add_scaled_sfs!(elbo_vars.elbo, elbo_vars.E_G, scale=-iota)
  println(elbo_vars.elbo.v)

  deepcopy(elbo_vars.elbo)
end

#their_accum = their_accum_pixel_ret();
@time our_accum = our_accum_pixel_ret();

@test_approx_eq their_accum.v our_accum.v


############################
############################
# Check each step


using Celeste
using CelesteTypes
using Base.Test
using SampleData
import Synthetic

include("src/ElboDeriv.jl")

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

ElboDeriv.get_expected_pixel_brightness!(
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


# Dig into accumulate_source_brightness!

# E[G] and E{G ^ 2} for a single source
E_G_s = elbo_vars.E_G_s;
E_G2_s = elbo_vars.E_G2_s;

clear_hessian = elbo_vars.calculate_hessian && elbo_vars.calculate_derivs
clear!(E_G_s, clear_hessian=clear_hessian)
clear!(E_G2_s, clear_hessian=clear_hessian);

@time a = mp.vp[s][ids.a]
@time fsm = (elbo_vars.fs0m_vec[s], elbo_vars.fs1m_vec[s]);
@time sb = sbs[s];

active_source = (s in mp.active_sources)

for i in 1:Ia # Stars and galaxies
  @time lf = sb.E_l_a[b, i].v * fsm[i].v
  @time llff = sb.E_ll_a[b, i].v * fsm[i].v^2

  @time E_G_s.v += a[i] * lf
  @time E_G2_s.v += a[i] * llff

  # Only calculate derivatives for active sources.
  if active_source && elbo_vars.calculate_derivs
    ######################
    # Gradients.

    @time E_G_s.d[ids.a[i], 1] += lf
    @time E_G2_s.d[ids.a[i], 1] += llff

    @time p0_shape = shape_standard_alignment[i]
    @time p0_bright = brightness_standard_alignment[i]
    @time u_ind = i == 1 ? star_ids.u : gal_ids.u

    # Derivatives with respect to the spatial parameters
    #a_fd = a[i] * fsm[i].d[:, 1]
    for p0_shape_ind in 1:length(p0_shape)
      @time E_G_s.d[p0_shape[p0_shape_ind], 1] +=
        sb.E_l_a[b, i].v * a[i] * fsm[i].d[p0_shape_ind, 1]
      @time E_G2_s.d[p0_shape[p0_shape_ind], 1] +=
        sb.E_ll_a[b, i].v * 2 * fsm[i].v * a[i] * fsm[i].d[p0_shape_ind, 1]
    end

    # Derivatives with respect to the brightness parameters.
    for p0_bright_ind in 1:length(p0_bright)
      @time E_G_s.d[p0_bright[p0_bright_ind], 1] +=
        a[i] * fsm[i].v * sb.E_l_a[b, i].d[p0_bright_ind, 1]
      @time E_G2_s.d[p0_bright[p0_bright_ind], 1] +=
        a[i] * (fsm[i].v^2) * sb.E_ll_a[b, i].d[p0_bright_ind, 1]
    end

    if elbo_vars.calculate_hessian
      ######################
      # Hessians.
      println("Hessian")
      # Data structures to accumulate certain submatrices of the Hessian.
      @time E_G_s_hsub = elbo_vars.E_G_s_hsub_vec[i];
      @time E_G2_s_hsub = elbo_vars.E_G2_s_hsub_vec[i];

      # The (a, a) block of the hessian is zero.

      # The (bright, bright) block:
      for p0_ind1 in 1:length(p0_bright), p0_ind2 in 1:length(p0_bright)
        # TODO: time consuming **************
        @time E_G_s.h[p0_bright[p0_ind1], p0_bright[p0_ind2]] =
          a[i] * fsm[i].v * sb.E_l_a[b, i].h[p0_ind1, p0_ind2]
        @time E_G2_s.h[p0_bright[p0_ind1], p0_bright[p0_ind2]] =
          a[i] * (fsm[i].v^2) * sb.E_ll_a[b, i].h[p0_ind1, p0_ind2]
      end

      # The (shape, shape) block:
      println("shape shape:")
      @time E_G_s_hsub.shape_shape = a[i] * sb.E_l_a[b, i].v * fsm[i].h;

      @time p1, p2 = size(E_G_s_hsub.shape_shape);
      for ind1 = 1:p1, ind2 = 1:p2
        @time E_G2_s_hsub.shape_shape[ind1, ind2] =
          2 * a[i] * sb.E_ll_a[b, i].v * (
            fsm[i].v * fsm[i].h[ind1, ind2] +
            fsm[i].d[ind1, 1] * fsm[i].d[ind2, 1])
      end

      # The u_u submatrix of this assignment will be overwritten after
      # the loop.
      for p0_ind1 in 1:length(p0_shape), p0_ind2 in 1:length(p0_shape)
        @time E_G_s.h[p0_shape[p0_ind1], p0_shape[p0_ind2]] =
          a[i] * sb.E_l_a[b, i].v * fsm[i].h[p0_ind1, p0_ind2]
        @time E_G2_s.h[p0_shape[p0_ind1], p0_shape[p0_ind2]] =
          E_G2_s_hsub.shape_shape[p0_ind1, p0_ind2];
      end

      # Since the u_u submatrix is not disjoint between different i, accumulate
      # it separate and add it at the end.
      @time E_G_s_hsub.u_u = E_G_s_hsub.shape_shape[u_ind, u_ind];
      @time E_G2_s_hsub.u_u = E_G2_s_hsub.shape_shape[u_ind, u_ind];

      # All other terms are disjoint between different i and don't involve
      # addition, so we can just assign their values (which is efficient in
      # native julia).

      # The (a, bright) blocks:
      for p0_ind in 1:length(p0_bright)
        @time E_G_s.h[p0_bright[p0_ind], ids.a[i]] =
          fsm[i].v * sb.E_l_a[b, i].d[p0_ind, 1]
        @time E_G2_s.h[p0_bright[p0_ind], ids.a[i]] =
          (fsm[i].v ^ 2) * sb.E_ll_a[b, i].d[p0_ind, 1]
      end
      @time E_G2_s.h[ids.a[i], p0_bright] = E_G2_s.h[p0_bright, ids.a[i]]';
      @time E_G_s.h[ids.a[i], p0_bright] = E_G_s.h[p0_bright, ids.a[i]]';

      # The (a, shape) blocks.
      for p0_ind in 1:length(p0_shape)
        @time E_G_s.h[p0_shape[p0_ind], ids.a[i]] =
          sb.E_l_a[b, i].v * fsm[i].d[p0_ind, 1]
        @time E_G2_s.h[p0_shape[p0_ind], ids.a[i]] =
          sb.E_ll_a[b, i].v * 2 * fsm[i].v * fsm[i].d[p0_ind, 1]
      end
      @time E_G2_s.h[ids.a[i], p0_shape] = E_G2_s.h[p0_shape, ids.a[i]]';
      @time E_G_s.h[ids.a[i], p0_shape] = E_G_s.h[p0_shape, ids.a[i]]';

      for ind_b in 1:length(p0_bright), ind_s in 1:length(p0_shape)
        @time E_G_s_hsub.bright_shape[ind_b, ind_s] =
          a[i] * sb.E_l_a[b, i].d[ind_b, 1] * fsm[i].d[ind_s, 1]
        @time E_G2_s_hsub.bright_shape[ind_b, ind_s] =
          2 * a[i] * sb.E_ll_a[b, i].d[ind_b, 1] * fsm[i].v * fsm[i].d[ind_s]
      end

      @time E_G_s.h[p0_bright, p0_shape] = E_G_s_hsub.bright_shape;
      @time E_G_s.h[p0_shape, p0_bright] = E_G_s_hsub.bright_shape';
      @time E_G2_s.h[p0_bright, p0_shape] = E_G2_s_hsub.bright_shape;
      @time E_G2_s.h[p0_shape, p0_bright] = E_G2_s_hsub.bright_shape';
    end # if calculate hessian
  end # if calculate derivatives
end # i loop

if elbo_vars.calculate_hessian
  # Accumulate the u Hessian.  u is the only parameter that is shared between
  # different values of i.
  # E_G_u_u_hess = zeros(2, 2);
  # E_G2_u_u_hess = zeros(2, 2);

  # For each value in 1:Ia, written this way for speed.
  @assert Ia == 2
  @time begin
    E_G_u_u_hess =
      elbo_vars.E_G_s_hsub_vec[1].u_u +
      elbo_vars.E_G_s_hsub_vec[2].u_u
  end

  @time begin
     E_G2_u_u_hess =
      elbo_vars.E_G2_s_hsub_vec[1].u_u +
      elbo_vars.E_G2_s_hsub_vec[2].u_u
  end

  # for i = 1:Ia
  #   E_G_u_u_hess += elbo_vars.E_G_s_hsub_vec[i].u_u
  #   E_G2_u_u_hess += elbo_vars.E_G2_s_hsub_vec[i].u_u
  # end
  @time E_G_s.h[ids.u, ids.u] = E_G_u_u_hess;
  @time E_G2_s.h[ids.u, ids.u] = E_G2_u_u_hess;
end

@time ElboDeriv.calculate_var_G_s!(elbo_vars, active_source)
