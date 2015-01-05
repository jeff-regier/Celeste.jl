#!/usr/bin/env julia

using Celeste
using CelesteTypes

function small_image_profile()
	srand(1)
	blob0 = StampBlob.load_stamp_blob(ENV["STAMP"], "164.4311-39.0359")
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
	mp = ModelInit.peak_init(blob, patch_radius=35., tile_width=2)
	elbo = ElboDeriv.elbo(blob, mp)
end


Profile.init(10^7, 0.001)
small_image_profile()
@profile small_image_profile()
#Profile.print()
Profile.print(format=:flat)


