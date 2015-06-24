# Based on util/test-psf.py in astrometry.net:
from astrometry.util import sdss_psf
from astrometry.sdss import *
from numpy import *


# Note that pyfits insists on underscores, not hyphens.
ps_filename = '/tmp/psField_003900_6_0269.fit'
psfield = pyfits.open(ps_filename)

x = 1024.0 - 1
y = 744.5 - 1

band = 0
raw_psf = sdss_psf.sdss_psf_at_points(psfield[band + 1], x, y)
raw_psf[0].max()

###################
# The raw code:
hdu = psfield[band + 1]

rtnscalar = True
x = atleast_1d(x)
y = atleast_1d(y)

psf = table_fields(hdu.data)

psfimgs = None
(outh, outw) = (None,None)

# From the IDL docs:
# http://photo.astro.princeton.edu/photoop_doc.html#SDSS_PSF_RECON
#   acoeff_k = SUM_i{ SUM_j{ (0.001*ROWC)^i * (0.001*COLC)^j * C_k_ij } }
#   psfimage = SUM_k{ acoeff_k * RROWS_k }

k = 0
# for k in range(len(psf)):
nrb = psf.nrow_b[k]
ncb = psf.ncol_b[k]

c = psf.c[k].reshape(5, 5)
c = c[:nrb,:ncb]

(gridi,gridj) = meshgrid(range(nrb), range(ncb))

# rrows is the vectorized image.
i = 0
xi = x
yi = y
psfimgs = [ zeros_like(psf.rrows[k]) ]
(outh,outw) = (psf.rnrow[k], psf.rncol[k])
for k in range(len(psf)):
	nrb = psf.nrow_b[k]
	ncb = psf.ncol_b[k]
	c = psf.c[k].reshape(5, 5)
	c = c[:nrb,:ncb]
	(gridi,gridj) = meshgrid(range(nrb), range(ncb))
	acoeff_k = sum(((0.001 * xi)**gridi * (0.001 * yi)**gridj * c))
	if True: # DEBUG
		print '--------------------------\n'
		print 'coeffs:', (0.001 * xi)**gridi * (0.001 * yi)**gridj
		print 'c:', c
		for (coi,ci) in zip(((0.001 * xi)**gridi * (0.001 * yi)**gridj).ravel(), c.ravel()):
			print 'co %g, c %g' % (coi,ci)
		print 'acoeff_k', acoeff_k
	this_image = acoeff_k * psf.rrows[k]
	psfimgs[i] += this_image
	print sum(this_image)

psfimgs = [img.reshape((outh,outw)) for img in psfimgs]
savetxt("/tmp/py_psf.csv", psfimgs[0], delimiter=",")


if rtnscalar:
	return psfimgs[0]
return psfimgs

for k in range(4):
	savetxt("/tmp/rrows_%d.csv" % k, psf.rrows[k], delimiter=",")
