# Load a correctly processed image from tractor.

from tractor.sdss import *
import argparse
import numpy

parser = argparse.ArgumentParser()

parser.add_argument('--run', type=int, help='The run number.')
parser.add_argument('--camcol', type=int, help='The camcol number.')
parser.add_argument('--field', type=int, help='The field number.')
parser.add_argument('--band', type=int, help='The band (a number from 1 to 5).')
parser.add_argument('--destination_base', type=str,
	                help='Output will be written to files like <destination_base>_<run>_<camcol>_<field>_<description>.csv')

args = parser.parse_args()

if not args.run:
	# Assume we're in interacive mode.
	args.run = 3900
	args.camcol = 6
	args.field = 269
	args.destination_base = '/tmp/test'
	args.band = 1

img = get_tractor_image(args.run, args.camcol, args.field, args.band, nanomaggies=True)
sources = get_tractor_sources_dr9(args.run, args.camcol, args.field,
	                              nanomaggies=True, fixedComposites=True, useObjcType=True)

file_base = args.destination_base + ('_%d_%d_%d_' % (args.run, args.camcol, args.field))
band_str = '%d_' % args.band
numpy.savetxt(file_base + band_str + "img.csv", img[0].data, delimiter=",")
numpy.savetxt(file_base + band_str + "psf.csv", img[0].psf, delimiter=",")