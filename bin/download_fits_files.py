#!/usr/bin/python
#
# Download the FITS files necessary to run Celeste for a
# particular run, camcol, and field.  Requires tractor:
# https://github.com/dstndstn/tractor

from tractor.sdss import *
import argparse
import urllib
import os
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--run', type=int,
	help='The run number.', required=True)
parser.add_argument('--camcol', type=int,
	help='The camcol number.', required=True)
parser.add_argument('--field', type=int,
	help='The field number.', required=True)
parser.add_argument('--destination_dir', type=str,
	help='Destination directory', required=True)

args = parser.parse_args()

sdss = DR8()

for file_type in ['photoObj', 'photoField', 'psField']:
	dest_url = sdss.get_url(file_type, args.run, args.camcol, args.field)
	print 'Downloading %s from %s' % (file_type, dest_url)
	urllib.urlretrieve(dest_url,
	                   os.path.join(args.destination_dir, basename(dest_url)))

bands = ['u', 'g', 'r', 'i', 'z']
for file_type in ['frame', 'fpM']:
	for band_letter in bands:
		dest_url = sdss.get_url(file_type, args.run, args.camcol,
		                        args.field, band=band_letter)
		print 'Downloading %s band %s from %s' % (file_type, band_letter, dest_url)
		dest_file = os.path.join(args.destination_dir, basename(dest_url))
		urllib.urlretrieve(dest_url, dest_file)
		# Some files need to be uncompressed.
		extension = os.path.splitext(basename(dest_url))[1]
		if extension == '.gz':
			print 'Gunzipping %s' % dest_file
			subprocess.call(['gunzip', dest_file])
		elif extension == '.bz2':
			print 'Bunzipping %s' % dest_file
			subprocess.call(['bunzip2', dest_file])
