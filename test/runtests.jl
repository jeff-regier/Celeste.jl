#!/usr/bin/env julia

using Celeste
using CelesteTypes
using Base.Test

const stamp_dir = joinpath(Pkg.dir("Celeste"), "dat")


function stamp_test()
	srand(1)
	blob0 = StampBlob.load_stamp_blob(stamp_dir, "164.4311-39.0359")

	brightness7000K = real(Planck.photons_expected(7000., 5., 1e4))

	two_bodies = [
		CatalogStar([11.1, 21.2], brightness7000K),
		CatalogGalaxy([15.3, 31.4], brightness7000K , 0.1, [6, 0., 6.]),
	]

   	blob = Synthetic.gen_blob(blob0, two_bodies)
	mp = ModelInit.peak_init(blob)
	@test mp.S == 2

	elbo = ElboDeriv.elbo(blob, mp)
	@test_approx_eq elbo.v -2.503656440951526e6
	truth1 = [-230903.60490336572,21299.08660612206,51088.55474253761,
		0.1614956164650373,2.0492672411811497,3.331005130357673,
		2.925751746716208,0.05621522390886591,0.0,0.2139999010600101,
		1.944472146778113,3.1704773266683755,2.8906675553297165,
		0.09169372979580104,35625.58719162395,-50221.120149363735,
		3497.9309989329117,-50812.5986990325]
	truth2 = [9908.365002397146,10482.295322197082,13140.007020472252,
		-0.16114679061138232,0.556809801077656,0.9819626050294983,
		0.9101889930241706,-0.17498922700435343,0.0,-0.07336958751764544,
		0.6557627595867292,1.120386573430293,1.071348038526756,
		-0.12507926751994441,4281.438872487195,-5146.3088665900195,
		2342.500310199912,-399.80421574499985]
	for p in 1:18
		@test_approx_eq elbo.d[p, 1] truth1[p]
		@test_approx_eq elbo.d[p, 2] truth2[p]
	end

	println("--- starting optimization---")
	OptimizeElbo.maximize_elbo(blob, mp)
	@test_approx_eq mp.vp[1].chi 0.0001
	@test_approx_eq mp.vp[2].chi 0.9999
	@test_approx_eq_eps mp.vp[1].mu[1] 11.1 0.05
	@test_approx_eq_eps mp.vp[1].mu[2] 21.2 0.05
	@test_approx_eq_eps mp.vp[2].mu[1] 15.3 0.05
	@test_approx_eq_eps mp.vp[2].mu[2] 31.4 0.05
	@test_approx_eq_eps mp.vp[2].Xi[1] 6. 0.05
	@test_approx_eq_eps mp.vp[2].Xi[2] 0. 0.05
	@test_approx_eq_eps mp.vp[2].Xi[3] 6. 0.05
end


function small_image_test()
	srand(1)
	blob0 = StampBlob.load_stamp_blob(stamp_dir, "164.4311-39.0359")
	for b in 1:5
		blob0[b].H, blob0[b].W = 100, 200
	end

	brightness7000K = real(Planck.photons_expected(7000., 10., 1e4))

	three_bodies = [
		CatalogStar([10.1, 12.2], brightness7000K),
		CatalogGalaxy([71.3, 100.4], brightness7000K , 0.1, [6, 0., 6.]),
		CatalogGalaxy([81.5, 103.6], brightness7000K , 0.1, [6, 0., 6.]),
	]

   	blob = Synthetic.gen_blob(blob0, three_bodies)
	mp = ModelInit.peak_init(blob)
	@test mp.S == 3

	elbo = ElboDeriv.elbo(blob, mp)
	@test_approx_eq elbo.v -1.0539564589332629e7
end



function tiling_test()
	srand(1)
	blob0 = StampBlob.load_stamp_blob(stamp_dir, "164.4311-39.0359")
	for b in 1:5
		blob0[b].H, blob0[b].W = 112, 238
	end

	brightness7000K = real(Planck.photons_expected(7000., 10., 1e4))

	three_bodies = [
		CatalogGalaxy([4.5, 3.6], brightness7000K , 0.1, [6, 0., 6.]),
		CatalogStar([60.1, 82.2], brightness7000K),
		CatalogGalaxy([71.3, 100.4], brightness7000K , 0.1, [6, 0., 6.]),
	]

   	blob = Synthetic.gen_blob(blob0, three_bodies)

	mp = ModelInit.cat_init(three_bodies)
	@test mp.S == 3

	elbo = ElboDeriv.elbo(blob, mp)

	@test_approx_eq elbo.v -9.363866983005373e6

	truth = [-509250.41464684,188.92437906,25251.46152459,0.65012746,
		1.32410644,2.06266958,6.02071596,0.070201,0.0,0.68068553,1.42667276,
		2.12416342,5.82810905,0.13986853,26693.38441889,-106792.13491265,
		347.13519247,-23517.62927772]
	for i in 1:18
		@test_approx_eq_eps elbo.d[i, 1 + i % 3] truth[i] 1e-6
	end
end


tiling_test()
stamp_test()
small_image_test()

