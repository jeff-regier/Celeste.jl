# Based on util/test-psf.py in astrometry.net:
import argparse

from astrometry.util import sdss_psf
from astrometry.sdss import *
from numpy import *

parser = argparse.ArgumentParser()

parser.add_argument('--ps_file', type=str,
	                help=('The psField file to read (no hyphens, only underscores are '
	                	  'allowed in the filename for some reason).'))
parser.add_argument('--band', type=int, help='The band (a number from 0 to 4).')
parser.add_argument('--row', type=float, help='The row at which to evaluate the psf (0-indexed).')
parser.add_argument('--col', type=float, help='The column at which to evaluate the psf (0-indexed).')
parser.add_argument('--destination_file', type=str, help='The name of the output csv file.')


args = parser.parse_args()

if not args.ps_file:
	# Assume we're in interacive mode.
	args.ps_file = '/tmp/psField_003900_6_0269.fit'
	args.band = 0
	args.col = 1024.0 - 1
	args.row = 744.5 - 1
	args.destination_file = '/tmp/py_psf.csv'

# Note that pyfits insists on underscores, not hyphens.
psfield = pyfits.open(args.ps_file)

print 'Reading psf:'
raw_psf = sdss_psf.sdss_psf_at_points(psfield[args.band + 1], args.row, args.col)
savetxt(args.destination_file, raw_psf, delimiter=",")
print 'Done.  Output written to %s' % args.destination_file