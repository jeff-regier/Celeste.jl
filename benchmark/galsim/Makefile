OUTPUT_DIR := output
TEST_IMAGES_BASE_URL := http://portal.nersc.gov/project/dasrepo/celeste/galsim_fits
LATEST_FITS_FILENAME := $(shell cat latest_filenames/latest_galsim_benchmarks.txt)
LATEST_FITS_PATH := $(OUTPUT_DIR)/$(LATEST_FITS_FILENAME)

default: benchmark

fetch: $(LATEST_FITS_PATH)

benchmark: GalsimBenchmark.jl run_galsim_benchmark.jl $(LATEST_FITS_PATH)
	julia run_galsim_benchmark.jl

$(LATEST_FITS_PATH):
	echo "Fetching GalSim test images from $(TEST_IMAGES_BASE_URL)/$(LATEST_FITS_FILENAME)"
	mkdir -p $(OUTPUT_DIR)
	curl --fail -o $(LATEST_FITS_PATH) $(TEST_IMAGES_BASE_URL)/$(LATEST_FITS_FILENAME)

generate_test_images: Vagrantfile bootstrap.sh generate_test_image.py test_case_definitions.py
	vagrant up
	vagrant ssh -c 'cd /vagrant && python generate_test_image.py'
	vagrant halt
