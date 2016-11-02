OUTPUT_DIR := output
TEST_IMAGES_FITS := $(OUTPUT_DIR)/galsim_test_images.fits
TEST_IMAGES_URL := http://portal.nersc.gov/project/dasrepo/celeste/galsim_test_images.fits

default: benchmark

benchmark: GalsimBenchmark.jl run_galsim_benchmark.jl $(TEST_IMAGES_FITS)
	julia run_galsim_benchmark.jl

$(TEST_IMAGES_FITS):
	mkdir -p $(dir $@)
	curl -o $@ $(TEST_IMAGES_URL)

generate_test_images: Vagrantfile bootstrap.sh generate_test_image.py galsim_truth.csv
	vagrant up
	vagrant ssh -c 'cd /vagrant && python generate_test_image.py'
	vagrant halt
