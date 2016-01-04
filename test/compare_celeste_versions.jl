# Compare the new Celeste version to an old version in another repo.

# Consider julia --track-allocation=user

using Celeste
using CelesteTypes
using Base.Test
using SampleData
import Synthetic

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

Profile.clear_malloc_data()
Profile.clear()
@profile ElboDeriv.elbo_likelihood(tiled_blob, mp);
#Profile.print()

# To see memory, quit and run
using Coverage
res = analyze_malloc("src");
res[(end - 20):end]



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
@time our_sbs = ElboDeriv.load_source_brightnesses(mp, true);

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
  sbs = ElboDeriv.load_source_brightnesses(mp, true);
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
  sbs = ElboDeriv.load_source_brightnesses(mp, true);
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
  sbs = ElboDeriv.load_source_brightnesses(mp, true);
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







################# Experimenting with BLAS

n = 1000
a = rand(n);
b = rand(n);
c = zeros(n, n);

@time BLAS.gemm!('N', 'T', 1.0, a, b, 1.0, c);
@time c[:,:] = a * b';


a = rand(n, 2);
b = rand(n, 2);
c = zeros(2 * n, 2 * n);
@time BLAS.ger!(1.0, a[:], b[:], c);
cblas = deepcopy(c);

# A little faster.
@time af = a[:]; bf = b[:];
c = zeros(2 * n, 2 * n);
@time BLAS.ger!(1.0, af, bf, c);

c = zeros(2 * n, 2 * n);
@time c = a[:] * b[:]';
cjul = deepcopy(c);
@test_approx_eq cjul cblas;

n = 3000
r1 = rand(n, n);

function f1!(r0, r)
  r0[:, :] = r + r0
end

function f2!(r0, r)
  for i1 = 1:n, i2 = 1:n
    r0[i1, i2] = r[i1, i2] + r0[i1, i2]
  end
end

function f3!(r0, r)
  r0 += r
end

function f4!(r0, r)
  r0[:, :] = r[:, :] + r0[:, :]
end


r0 = ones(n, n);
@time f1!(r0, r1);

# So slow
# r0 = ones(n, n);
# @time f2!(r0, r1);

r0 = ones(n, n);
@time f3!(r0, r1);

r0 = ones(n, n);
@time f4!(r0, r1);


using Base.Test
n = 1000;
x = rand(n, n);
y = zeros(n, n);
@time BLAS.blascopy!(n * n, x, 1, y, 1);
@test_approx_eq x y

@time x = deepcopy(y);


n = 1000;
y = rand(2 * n, 2 * n);
y_sub = y[1:n, 1:n];

x = zeros(n, n);
@time x[:, :] += 3.0 * y[1:n, 1:n];
x = zeros(n, n);
@time BLAS.axpy!(3.0, y[1:n, 1:n], x);
x = zeros(n, n);
@time BLAS.axpy!(3.0, y_sub, x);

######
n = 100;
x = rand(n, n);
sub_ind = 2 * (1:n)
sub_ind_col = collect(sub_ind);

y = zeros(2 * n, 2 * n);
@time y[sub_ind, sub_ind] += 3.0 * x;
y = zeros(2 * n, 2 * n);

# These each allocate less memory as you go down.
@time y[sub_ind_col, sub_ind_col] += 3.0 * x;
@time y[sub_ind_col, sub_ind_col] = 3.0 * x;
@time y[sub_ind, sub_ind] = x;
@time y[sub_ind_col, sub_ind_col] = x;

# Dig that this doesn't work.
y = zeros(2 * n, 2 * n);
@time BLAS.axpy!(3.0, x, y[sub_ind, sub_ind]);

# Neither does this.
y = zeros(2 * n, 2 * n);
y_sub = y[sub_ind, sub_ind];
@time BLAS.axpy!(3.0, x, y_sub);

# This allocates a ton of memory.
y = zeros(2 * n, 2 * n);
@time begin
  for i1 in 1:length(sub_ind), i2 in 1:length(sub_ind)
    j1 = sub_ind[i1]
    j2 = sub_ind[i2]
    z = 3 * x[i1, i2] + y[j1, j2]
    #y[j1, j2] += 3 * x[i1, i2]
    y[j1, j2] = z
  end
end
# n = 1000;   0.726820 seconds (7.47 M allocations: 129.318 MB, 12.70% gc time)
# n = 100;   0.005881 seconds (50.20 k allocations: 943.781 KB)

# This allocates a ton of memory.
y = zeros(2 * n, 2 * n);
@time begin
  for i in eachindex(x)
    i1, i2 = ind2sub(size(x), i)
    j1 = sub_ind[i1]
    j2 = sub_ind[i2]
    y[j1, j2] += 3 * x[i1, i2]
  end
end




n = 10000
y = ones(n);
x = rand(n);

@time begin
  for i1 = 1:n
    y[i1] += x[i1] * 3
  end
end

z = 0.0
@time begin
  for i1 = 1:n
    z = x[i1] * 3 + y[i1]
    y[i1] = z
  end
end

z = 0.0
@time begin
  for i1 = 1:n
    global z
    z = x[i1] * 3 + y[i1]
    y[i1] = z
  end
end






#
