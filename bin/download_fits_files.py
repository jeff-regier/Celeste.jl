# Load a processed image from tractor.

from tractor.sdss import *
import argparse
import urllib
#import numpy
#import copy


parser = argparse.ArgumentParser()

parser.add_argument('--run', type=int, help='The run number.')
parser.add_argument('--camcol', type=int, help='The camcol number.')
parser.add_argument('--field', type=int, help='The field number.')
#parser.add_argument('--band', type=int, help='The band (a number from 0 to 4).')
parser.add_argument('--destination_base', type=str,
	                help='Output will be written to files like <destination_base>_<run>_<camcol>_<field>_<description>.csv')

args = parser.parse_args()

if not args.run:
	# Assume we're in interacive mode.
	args.run = 3900
	args.camcol = 6
	args.field = 269
	args.destination_base = '/tmp/test'

sdss = DR8()

urllib.urlretrieve(sdss.get_url('photoObj', args.run, args.camcol, args.field),
                   '/tmp/photoObj.fits')
sdss.get_url('photoField', args.run, args.camcol, args.field)
sdss.get_url('psField', args.run, args.camcol, args.field)

bands = ['u', 'g', 'r', 'i', 'z']
for band_letter in bands:
	sdss.get_url('fpM', args.run, args.camcol, args.field, band=band_letter)
	sdss.get_url('frame', args.run, args.camcol, args.field, band=band_letter)
