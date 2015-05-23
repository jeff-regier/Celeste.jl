# Load a correctly processed image from tractor.

from tractor.sdss import *
import argparse
import numpy
import copy

parser = argparse.ArgumentParser()

parser.add_argument('--run', type=int, help='The run number.')
parser.add_argument('--camcol', type=int, help='The camcol number.')
parser.add_argument('--field', type=int, help='The field number.')
parser.add_argument('--band', type=int, help='The band (a number from 0 to 4).')
parser.add_argument('--destination_base', type=str,
	                help='Output will be written to files like <destination_base>_<run>_<camcol>_<field>_<description>.csv')

args = parser.parse_args()

if not args.run:
	# Assume we're in interacive mode.
	args.run = 3900
	args.camcol = 6
	args.field = 269
	args.destination_base = '/tmp/test'
	args.band = 0

bands = ['u', 'g', 'r', 'i', 'z']
bandname = bands[args.band]

# Very useful:
#print ''.join(inspect.getsourcelines(pyfits.HDUList)[0])


img = get_tractor_image(args.run, args.camcol, args.field, args.band, nanomaggies=True)
sources = get_tractor_sources_dr9(args.run, args.camcol, args.field,
	                              nanomaggies=True, fixedComposites=True, useObjcType=True)


file_base = args.destination_base + ('_%d_%d_%d_' % (args.run, args.camcol, args.field))
band_str = '%d_' % (args.band + 1) # Python uses 0 indexing
numpy.savetxt(file_base + band_str + "img.csv", img[0].data, delimiter=",")
numpy.savetxt(file_base + band_str + "psf.csv", img[0].psf, delimiter=",")



# Later data releases are not supported by get_tractor_image.
sdss = DR8()

# This doesn't work:
#sdss.get_url('fpC', args.run, args.camcol, args.field, 'r')
#sdss.get_url('fpM', args.run, args.camcol, args.field, 'r')

# Mask the image.
fpM = sdss.readFpM(args.run, args.camcol, args.field, bandname)
for plane in [ 'INTERP', 'SATUR', 'CR', 'GHOST' ]:
	masked_img_data = copy.deepcopy(img[0].data)
	print plane
	fpM.setMaskedPixels(plane, masked_img_data, NaN)
	print sum(numpy.isnan(masked_img_data))

numpy.savetxt(file_base + band_str + "masked_img.csv", masked_img_data, delimiter=",")


# This doesn't match up with Julia, because the files seem to have different data.
# For example:
size(fpM.getMaskPlane('INTERP').rmin) # INTERP is element 1 in the 0-indexed python.
# == 168
# In Julia:
# read(fpm_fits[2], "rmax") == 188
# There are other data in Julia that are missing from tractor.


