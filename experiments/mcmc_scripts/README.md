# celeste-stats-proj
scripts/plots/docs for celeste modeling + inference project

AIS-MCMC Results Roadmap

0. **Generate Data; Run VB**

   - (a)  Generate s82 coadd + primary csv catalogs, will be stored in directory `Celeste.jl/benchmark/accuracy/output`
     ```
     $ cd Celeste.jl/benchmark/accuracy
     $ julia write_ground_truth_catalog_csv.jl coadd
     $ julia sdss_rcf_to_csv.jl  # DEFAULTS to S82 Field
     $ julia run_celeste_on_field.jl --use-full-initialization --initialization-catalog output/sdss_4263_5_119_primary_<hash>.csv
     ```
     This will create a file `sdss_4263_5_119_prediction_<hash>.csv` in `Celeste.jl/benchmark/accuracy/output`. 
  
   - (b) Also generate Synthetic catalog and images
     ```
     $ julia write_ground_truth_catalog_csv.jl prior
     $ julia generate_synthetic_field.jl prior_<hash>.csv
     $ julia run_celeste_on_field.jl --initialization-catalog output/prior_<hash>.csv --images-jld output/prior_<hash>_synthetic.jld
     ```
     which will create the jld images in directory `Celeste.jl/benchmark/accuracy/output`, and use them for inference.
     Note that I did not trigger the flag `--use-full-initialization`, as the initialization in the synthetic case 
     is the ground truth.
     
   Note: on cori, an interactive queue can be created by running
   ```
   $ salloc -N 1 -C haswell --qos=interactive -t 02:00:00 -A dasrepo
   ```

1. **Run AIS-MCMC Code** 

   First, edit `run_s82_shards.sh` and `run_shards.sh` so that they point to the correct files generated in step 0.

   - (a) Run on field, s82
     ```
     $ cd celeste-stats-proj/src/
     $ ./run_s82_shards.sh
     ```
     will compute AIS-MCMC samples for each source in Stripe 82., storing them in directory `ais-output-s82`.
     Note that `use_robust_likelihood` is an argument that can be toggled in the

   - (b) run on synthetic field
     ```
     $ cd celeste-stats-proj/src/
     $ ./run_shards.sh
     ```
     storing samples in directory `ais-output-synthetic`. 


2. **Score Results**

   For both stripe82 and synthetic data, run the following scripts

   - (a) Create output CSVs (to be used for python plots) 
     ```
     $ ./score_mcmc_results.jl --ais-output <ais-output-dir> --output-dir <data-frame-output-dir> \\
       --truth-csv <> --vb-csv <> --photo-csv <>
     ```
     on the output of steps 0, and 1 to create results data farames. 
     
      ```
      julia score_mcmc_results.jl --ais-output ais-output-s82 --output-dir <output_location> --truth-csv coadd_for<field+hash>.csv --vb-csv sdss_<field>_predictions_<hash>.csv --photo-csv sdss_<field>_primary_<hash>.csv
      ```

      For example --- the following works on `cori`

      ```
      julia score_mcmc_results.jl --ais-output ais-output-s82 --output-dir s82-results-robust \
        --truth-csv ~/Proj/Celeste.jl/benchmark/accuracy/output/coadd_for_4263_5_119_d1aa0324ac.csv \
        --vb-csv ~/Proj/Celeste.jl/benchmark/accuracy/output/sdss_4263_5_119_predictions_3ad3e1db41.csv \
        --photo-csv ~/Proj/Celeste.jl/benchmark/accuracy/output/sdss_4263_5_119_primary_a1b7c2f263.csv
      ```

   - (b) Locally (or wherever matplotlib can run) run
      ```
      $ python make_mcmc_results_figure.py --results-dir <results-dir>
      ```

      To make the results figures, including

        - QQ plot 
        - Star/Gal ROC Curves + AUC comparison
        - VB vs MCMC Uncertainty Comparison Plots
        - Per Parameter Error Comparison (VB vs MCMC vs Photo) plots
