#!/usr/bin/env python

import sys

# sewpy is available at https://github.com/megalut/sewpy
import sewpy

#import logging
#logging.basicConfig(format='%(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)


sew = sewpy.SEW(
		params=["NUMBER", "EXT_NUMBER", "X_WORLD", "Y_WORLD", "FLUX_APER(3)", "FLUX_BEST", "MAG_APER", "CLASS_STAR", "FLAGS"],
		config={"DETECT_MINAREA":10, "PHOT_APERTURES":"5, 10, 20", "PIXEL_SCALE":0},
		sexpath="sextractor"
	)

out = sew(sys.argv[1])

print(out["table"]) # This is an astropy table.

