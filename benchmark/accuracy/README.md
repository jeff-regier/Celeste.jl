# Accuracy benchmarks

The code in this directory and the module `src/AccuracyBenchmark.jl` support a variety of possible
benchmarks of Celeste's accuracy.

## Overview

Each accuracy benchmark consists of running Celeste on a single field images (five images
techincally, one for each band) and comparing inferred parameters for all sources present to known
"ground truth" values. Here are the components of an accuracy benchmark:

1. We start with a ground truth catalog.
2. We get a set of band images corresponding to this catalog.
3. We run Celeste on these images, with particular initialization values corresponding to the given
   images, generating a predition catalog.
4. We compare one or more prediction catalogs to the ground truth, summarizing accuracy for each
   parameter.

All catalogs are stored in a common CSV format; see `AccuracyBenchmark.{read,write}_catalog()`.

At each step we have some choices:

## Ground truth catalog

The command

```
$ julia write_ground_truth_catalog_csv.jl [coadd|prior]
```

writes a ground truth catalog to the `output` subdirectory.

* `coadd` uses the SDSS Stripe82 "coadd" catalog, already (manually) pulled from the SDSS CasJobs
server (and checked into the Celeste repository under `test/data`).
* `prior` draws 500 random sources from the Celeste prior (with a few added prior distributions for
  parameters which don't have a prior specified in Celeste).

## Imagery

For Stripe82, one can use real SDSS imagery which has already been downloaded and checked in (under
`test/data`).

For any ground truth catalog, one can generate synthetic imagery in two ways:

* Using GalSim, with the `benchmark/galsim/galsim_field.py` script. See the README in that directory
  for more details (setting up GalSim is nontrivial). This process will generate a FITS file under
  `benchmark/galsim/output`.
* By drawing from the Celeste likelihood model. TODO: this has yet to be implemented.


## Reading the Stripe82 "primary" catalog

The command

```
$ julia sdss_rcf_to_csv.jl <coadd_catalog>
```

will read the Stripe82 "primary" catalog (from pre-downloaded FITS files) and write it in CSV form
to the `output` subdirectory. (The coadd catalog is used only for `objid` matching.) This is useful
for two things:

1. Initializing Celeste for a run on Stripe82 imagery.
2. Comparing Celeste's accuracy to the "primary" catalog.

## Running Celeste

The script `run_celeste_on_field.jl` will run Celeste on given imagery, with a given initialization
catalog, writing predictions to a new catalog under the `output` subdirectory.

* You can specify a FITS file containing imagery, or the default behavior is to read Stripe82 SDSS
  imagery.
* By default, Celeste will be initialized only with a noisy position, so you can pass a ground truth
  catalog for synthetic imagery without "cheating". Alternatively, if you pass
  `--use-full-initialization`, Celeste will be initialized with all information from the given
  catalog, so you can mimic how Celeste get initialized with the SDSS "primary" catalog in real
  runs.
* The script supports single or joint inference and MOG or FFT PSF models.

## Scoring accuracy

The command

```
$ julia benchmark/accuracy/score_predictions.jl <ground truth CSV> <predictions CSV> [predictions CSV]
```

compares one or two prediction catalogs to a ground truth catalog, summarizing their performance
(and comparing to each other, if two are given).


## Examples

Here are some examples of use:

* To run Celeste on Stripe82 real imagery using "primary" predictions for initialization, as in real
  runs, and compare Celeste to Stripe82 primary accuracy:
    
    ```
    $ julia benchmark/accuracy/write_ground_truth_catalog_csv.jl coadd
    $ julia benchmark/accuracy/sdss_rcf_to_csv.jl \
        benchmark/accuracy/output/stripe82_coadd_catalog_<hash>.csv
    $ julia benchmark/accuracy/run_celeste_on_field.jl --use-full-initialization \
        benchmark/accuracy/output/sdss_4263_5_119_primary_<hash>.csv
    $ julia benchmark/accuracy/score_predictions \
        benchmark/accuracy/output/stripe82_coadd_catalog_<hash>.csv \
        benchmark/accuracy/output/sdss_4263_5_119_primary_<hash>.csv \
        benchmark/accuracy/output/sdss_4263_5_119_predictions_<hash>.csv
    ```
    
* To run Celeste on GalSim imagery from a "prior" ground truth catalog, using partial information
  from the ground truth catalog for initialization, and compare single to joint inference:
    
    ```
    $ julia benchmark/accuracy/write_ground_truth_catalog_csv.jl prior
    # go to benchmark/galsim/ and generate synthetic imagery from the above-generated catalog
    $ julia benchmark/accuracy/run_celeste_on_field.jl \
        benchmark/accuracy/output/celeste_prior_catalog_<hash>.csv
        --image-fits benchmark/galsim/output/celeste_prior_catalog_<hash>_images_<hash>.fits
    $ julia benchmark/accuracy/score_predictions \
        benchmark/accuracy/output/celeste_prior_catalog_<hash>.csv \
        benchmark/accuracy/output/celeste_prior_catalog_<hash>_images_<hash>_predictions_<hash>.csv
    ```
