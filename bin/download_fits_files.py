# Load a processed image from tractor.

from tractor.sdss import *
import argparse
import urllib

parser = argparse.ArgumentParser()

parser.add_argument('--run', type=int, help='The run number.')
parser.add_argument('--camcol', type=int, help='The camcol number.')
parser.add_argument('--field', type=int, help='The field number.')
parser.add_argument('--destination_dir', type=str, help='Destination directory')

args = parser.parse_args()

if not args.run:
	# Assume we're in interacive mode for debugging.
	args.run = 3900
	args.camcol = 6
	args.field = 269
	args.destination_dir = '/tmp/'

sdss = DR8()

for file_type in ['photoObj', 'photoField', 'psField']:
	dest_url = sdss.get_url(file_type, args.run, args.camcol, args.field)
	print 'Downloading %s from %s' % (file_type, dest_url)
	urllib.urlretrieve(dest_url,
	                   '%s%s' % (args.destination_dir, basename(dest_url)))

bands = ['u', 'g', 'r', 'i', 'z']
for file_type in ['frame', 'fpM']:
	for band_letter in bands:
		dest_url = sdss.get_url(file_type, args.run, args.camcol,
		                        args.field, band=band_letter)
		print 'Downloading %s band %s from %s' % (file_type, band_letter, dest_url)
		urllib.urlretrieve(dest_url,
		                   '%s%s' % (args.destination_dir, basename(dest_url)))
