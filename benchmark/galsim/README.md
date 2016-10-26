# GalSim benchmarks

The code in this directory uses [GalSim](https://github.com/GalSim-developers/GalSim) to generate
synthetic star and galaxy imagery and uses these images to benchmark Celeste's accuracy.

## Fetching test images

TODO: write a Makefile to fetch an uploaded FITS file from NERSC.

## Generating test images

If you wish the generate test images yourself, the script `generate_test_image.py` generates the
images using GalSim. If you have GalSim installed on your local Python you can run it
directly. Installing GalSim is a bit cumbersome, though, so you may wish to use the included
`Vagrantfile`:

1. Install [Vagrant](https://www.vagrantup.com/).
2. From this directory, run `vagrant up`. This will take a while the first time (downloading a
   machine image and installing GalSim, a process handled by `bootstrap.sh`) but should be less than
   a minute thereafter.
3. Run `vagrant ssh`. You will now be logged into the virtual machine with GalSim installed.
4. From within the VM, run `cd /vagrant && python generate_test_image.py`.

You will then have a file `output/galsim_test_images.fits` (under this directory) containing the
test images in a multi-extension FITS file.

The test cases are currently described in `galsim_truth.csv`, one line per test case. The generated
FITS files contains one image for each of the five color bands for each test case. The FITS header
on each extension includes all ground truth data.

## Running Celeste on test images

Once you have `output/galsim_test_images.fits`, use `julia run_galsim_benchmark.jl` to run Celeste
on each test case and report inferred and ground truth values.
