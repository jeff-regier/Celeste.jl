#!/usr/bin/env python

<<<<<<< HEAD
import sys

# sewpy is available at https://github.com/megalut/sewpy
import sewpy
=======
import sewpy
import sys
>>>>>>> making some progress on the galsim/sextractor notebook

#import logging
#logging.basicConfig(format='%(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)


sew = sewpy.SEW(
<<<<<<< HEAD
		params=["NUMBER", "EXT_NUMBER", "X_WORLD", "Y_WORLD", "FLUX_APER(3)", "FLUX_BEST", "MAG_APER", "CLASS_STAR", "FLAGS"],
		config={"DETECT_MINAREA":10, "PHOT_APERTURES":"5, 10, 20", "PIXEL_SCALE":0},
=======
		params=["X_IMAGE", "Y_IMAGE", "FLUX_APER(3)", "FLAGS"],
		config={"DETECT_MINAREA":10, "PHOT_APERTURES":"5, 10, 20"},
>>>>>>> making some progress on the galsim/sextractor notebook
		sexpath="sextractor"
	)

out = sew(sys.argv[1])

print(out["table"]) # This is an astropy table.

