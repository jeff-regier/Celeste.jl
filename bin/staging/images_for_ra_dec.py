#!/usr/bin/python
import argparse

from astrometry.sdss.fields import radec_to_sdss_rcf
from astrometry.sdss.fields import RaDecToRcf

parser = argparse.ArgumentParser()

parser.add_argument('--RA', type=float,
	help='The RA location.', required=True)
parser.add_argument('--DEC', type=float,
	help='The DEC location.', required=True)
parser.add_argument('--window_flist', type=str,
	help='A full path to the window_flist.fits file.',
	default='/home/rgiordan/Documents/git_repos/Celeste.jl/dat/window_flist.fits')
parser.add_argument('--download_command', type=str,
	help='A full path to the download_fits_files.py script.',
	default='/home/rgiordan/Documents/git_repos/Celeste.jl/bin/download_fits_files.py')

args = parser.parse_args()

RD = RaDecToRcf(tablefn=args.window_flist)
for image in RD(args.RA, args.DEC, spherematch=True, radius=0, contains=False):
	print "{command} --run={run} --camcol={camcol} --field={field}".format(
		command=args.download_command, run=image[0], camcol=image[1], field=image[2])
