#!/usr/bin/env python

import sewpy
import sys

#import logging
#logging.basicConfig(format='%(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)


sew = sewpy.SEW(
		params=["X_IMAGE", "Y_IMAGE", "FLUX_APER(3)", "FLAGS"],
		config={"DETECT_MINAREA":10, "PHOT_APERTURES":"5, 10, 20"},
		sexpath="sextractor"
	)

out = sew(sys.argv[1])

print(out["table"]) # This is an astropy table.

