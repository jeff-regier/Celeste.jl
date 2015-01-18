#!/usr/bin/env julia

using Celeste
using CelesteTypes

import Planck
import Synthetic


function small_image_profile()
	srand(1)
	blob0 = SDSS.load_stamp_blob(ENV["STAMP"], "164.4311-39.0359")
	for b in 1:5
		blob0[b].H, blob0[b].W = 2000, 2000
	end

	brightness7000K = real(Planck.photons_expected(7000., 10., 1e4))

	S = 300
	locations = rand(2, S) .* 2000.

	S_bodies = CatalogEntry[CatalogStar(locations[:, s][:], brightness7000K) for s in 1:S]

   	blob = Synthetic.gen_blob(blob0, S_bodies)
	mp = ModelInit.cat_init(S_bodies, patch_radius=20., tile_width=10)
	elbo = ElboDeriv.elbo(blob, mp)
end


Profile.init(10^7, 0.001)
small_image_profile()
@profile small_image_profile()
#Profile.print()
Profile.print(format=:flat)


