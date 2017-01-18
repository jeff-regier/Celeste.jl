OUTPUT_DIR := output
TEST_IMAGES_BASE_URL := http://portal.nersc.gov/project/dasrepo/celeste/galsim_fits
BENCHMARK_FITS_FILENAME := $(shell cat latest_filenames/latest_galsim_benchmarks.txt)
BENCHMARK_FITS_PATH := $(OUTPUT_DIR)/$(BENCHMARK_FITS_FILENAME)
FIELD_FITS_FILENAME := $(shell cat latest_filenames/latest_galsim_field.txt)
FIELD_FITS_PATH := $(OUTPUT_DIR)/$(FIELD_FITS_FILENAME)

default: benchmark

fetch: $(BENCHMARK_FITS_PATH)
fetch_field: $(FIELD_FITS_PATH)

benchmark: GalsimBenchmark.jl run_galsim_benchmark.jl $(BENCHMARK_FITS_PATH)
	julia run_galsim_benchmark.jl

field: GalsimBenchmark.jl run_galsim_field.jl $(FIELD_FITS_PATH)
	julia run_galsim_field.jl

$(BENCHMARK_FITS_PATH):
	echo "Fetching GalSim test images from $(TEST_IMAGES_BASE_URL)/$(BENCHMARK_FITS_FILENAME)"
	mkdir -p $(OUTPUT_DIR)
	curl --fail -o $(BENCHMARK_FITS_PATH) $(TEST_IMAGES_BASE_URL)/$(BENCHMARK_FITS_FILENAME)

$(FIELD_FITS_PATH):
	echo "Fetching GalSim test images from $(TEST_IMAGES_BASE_URL)/$(FIELD_FITS_FILENAME)"
	mkdir -p $(OUTPUT_DIR)
	curl --fail -o $(FIELD_FITS_PATH) $(TEST_IMAGES_BASE_URL)/$(FIELD_FITS_FILENAME)

generate_benchmark_fits: Vagrantfile bootstrap.sh generate_test_image.py galsim_benchmarks.py
	vagrant up
	vagrant ssh -c 'cd /vagrant && python galsim_benchmarks.py'
	vagrant halt

generate_field_fits: Vagrantfile bootstrap.sh generate_test_image.py galsim_field.py
	vagrant up
	vagrant ssh -c 'cd /vagrant && python galsim_field.py'
	vagrant halt
