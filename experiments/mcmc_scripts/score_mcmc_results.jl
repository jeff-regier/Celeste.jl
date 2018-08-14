#!/usr/bin/env julia

#######
# CLI #
#######
import Celeste.ArgumentParse
parser = Celeste.ArgumentParse.ArgumentParser()
ArgumentParse.add_argument(
    parser,
    "--ais-output",
    help="Directory containing source-specific AIS sample files",
)
ArgumentParse.add_argument(
    parser,
    "--output-dir",
    help="Directory to output results dataframes",
)
ArgumentParse.add_argument(
    parser,
    "--truth-csv",
    help="True file (coadd, or prior generated file)",
)
ArgumentParse.add_argument(
    parser,
    "--vb-csv",
    help="VB inference file for comparison",
)
ArgumentParse.add_argument(
    parser,
    "--photo-csv",
    help="Photo inference file for comparison"
)
parsed_args = ArgumentParse.parse_args(parser, ARGS)
println(parsed_args)
println("======= scoring MCMC results ============")


#######################
# Analysis Output Dir #
#######################
import JLD, CSV
using DataFrames, StatsBase
import Celeste: AccuracyBenchmark, Model, MCMC

if haskey(parsed_args, "output-dir")
  output_dir = parsed_args["output-dir"]
else
  output_dir = "./mcmc_result_dataframes"
end
println(" ... saving output to ", output_dir)
if !isdir(output_dir)
  mkdir(output_dir)
end


#######################################################################
# collect MCMC files and summarize them into a results dataframe      #
#######################################################################
if haskey(parsed_args, "ais-output")
    ais_output_dir = parsed_args["ais-output"]
else
    ais_output_dir = "ais-output/synthetic-run-10-31"
end
ais_out = filter!(r"\.jld$", readdir(ais_output_dir))
results = [JLD.load(joinpath(ais_output_dir, f))["res"] for f in ais_out];
results = [res for res in results if !res["is_sky_bad"]]
@printf " ... found %d results files in directory %s \n" length(results) ais_output_dir


#####################################################
# Compute prob star/gal (based on model evidence    #
#####################################################
function compute_pstar(res; prior_is_star=.28) #[.28, .72])
    # is_star = [.28, .72] vs [.999, .001] vs [.5, .5]
    type_chain  = zeros(length(res["gal_bootstrap"]))
    lnprob_a    = log(prior_is_star)
    lnprob_nota = log(1.-prior_is_star)
    for n in 1:length(type_chain)
        # normalizing constant is ln p(data | star)
        # so p(star | data) \propto p(data | star) p(star)
        lnprob_star = res["star_bootstrap"][n] + lnprob_a
        lnprob_gal  = res["gal_bootstrap"][n] + lnprob_nota
        lnsum = Model.logsumexp([lnprob_star, lnprob_gal]) # normalize
        type_chain[n] = lnprob_star - lnsum
    end
    ave_pstar = Model.logsumexp(type_chain) - log(length(type_chain))
    return ave_pstar
end

for res in results
  res["ave_pstar"] = compute_pstar(res; prior_is_star=.28)
end

# must correct for off-by-pixel value
star_summary, gal_summary, results_df = MCMC.consolidate_samples(results;
    summarize=median)
#results_df[:dec] -= .396 / 3600.
#results_df[:ra]  -= .397 / 3600.


######################################################################
# Compute the per chain R Hats (TODO put these in the alg itself)    #
######################################################################
function compute_ss_rhat(sdf; num_chains=25, num_samps_per_chain=25)
    chains = []
    si = 1
    for i in 1:num_chains
        endi = si + num_samps_per_chain - 1
        #@printf "chunked %d : %d \n" si endi
        push!(chains, Array(sdf[si:endi,10:10]))
        si += num_samps_per_chain
    end
    return MCMC.potential_scale_reduction_factor(chains)
end

star_psrf = mean(vcat([compute_ss_rhat(res["star_samples"])
                       for res in results]...), 2)
gal_psrf  = mean(vcat([compute_ss_rhat(res["gal_samples"])
                       for res in results]...), 2)


######################################################################
# Load comparison data frames                                        #
######################################################################
if haskey(parsed_args, "truth-csv")
  truth_file = parsed_args["truth-csv"]
else
  truth_file = expanduser("~/Proj/Celeste.jl/benchmark/accuracy/output/prior_6328dda483.csv")
end
println(" ... loaded truth dataframe from ", truth_file)
truth = AccuracyBenchmark.read_catalog(truth_file)

if haskey(parsed_args, "vb-csv")
  vb_file = parsed_args["vb-csv"]
else
  vb_file = expanduser("~/Proj/Celeste.jl/benchmark/accuracy/output/prior_6328dda483_synthetic_6c24c1c9ad_predictions_342302e66a.csv")
end
println(" ... loaded VB comparison DF from ", vb_file)
vbdf = AccuracyBenchmark.read_catalog(vb_file)

if haskey(parsed_args, "photo-csv")
  photo_file = parsed_args["photo-csv"]
else
  println(" ... no photo file passed in --- loading VB as dummy ")
  photo_file = vb_file
end
photodf = AccuracyBenchmark.read_catalog(photo_file)


############################################################################
# Match prediction dataframes first --- throw away the same sources as VB  #
############################################################################
matched_truth, matched_prediction_dfs = AccuracyBenchmark.match_catalogs(
    truth, [vbdf, results_df, photodf]) #, tol=.396/3600*4)
@printf " ... found %d matched sources for comparison \n" size(matched_truth, 1)
truth = matched_truth
vbdf = matched_prediction_dfs[1]
results_df = matched_prediction_dfs[2]
photodf = matched_prediction_dfs[3]
@printf "Found %d nan r fluxes in the VB dataframe\n" sum(isnan.(vbdf[:flux_r_nmgy]))

#bad_idx = isnan.(vbdf[:flux_r_nmgy])
#truth = truth[.~bad_idx,:]
#vbdf = vbdf[.~bad_idx,:]
#results_df = results_df[.~bad_idx,:]
#photodf = photodf[.~bad_idx,:]


############################################
# Compare MCMC results samples to truth    #
############################################
function get_percentile_df(truth, sample_list)

    STDERR_COLUMNS = Set([
        :log_flux_r_stderr,
        :color_ug_stderr,
        :color_gr_stderr,
        :color_ri_stderr,
        :color_iz_stderr,
    ])

    # store these values --- easier that way
    truth[:log_flux_r] = log.(truth[:flux_r_nmgy])

    # go through each truth source, match it to a sample in the results list
    _, _, sample_summary = MCMC.consolidate_samples(sample_list; summarize=mean)
    #sample_summary[:dec] -= .396 / 3600.
    #sample_summary[:ra]  -= .396 / 3600.
    samp_pos = convert(Array, sample_summary[[:ra, :dec]])

    function get_percentile_column(column_name)
        percentiles, names = []  , []
        for i in 1:size(truth, 1)

            # match this truth position to the appropriate set of samples
            pos_i = convert(Array, truth[[:ra, :dec]][i, :])
            dists = sum(broadcast(-, samp_pos, pos_i).^2, 2)
            near_dist, near_idx = findmin(dists)

            # grab samples of appropriate source type
            sdict = results[near_idx]
            pstar = exp(sdict["ave_pstar"])
            samps = (pstar > .5) ? sdict["star_samples"] : sdict["gal_samples"]

            # truth percentile
            true_val = truth[column_name][i]
            true_percentile = sum(true_val .> sort(samps[column_name])) / float(size(samps, 1))

            push!(percentiles, true_percentile)
            push!(names, column_name)
        end
        return percentiles, names
    end

    # assemble dataframe of percentile values
    column_names = [:log_flux_r, :color_ug, :color_gr, :color_ri, :color_iz]
    res = [get_percentile_column(c) for c in column_names]
    pers = vcat([r[1] for r in res]...)
    names = vcat([r[2] for r in res]...)
    return DataFrame([names, pers], [:name, :percentile])
end


###############################################
# Compute scores for comparison data frames   #
###############################################
scores = AccuracyBenchmark.score_predictions(truth, [vbdf, results_df])
println("Comparing vb (first) to MCMC (second)")
println(repr(scores))

# uncertainties
vb_udf = AccuracyBenchmark.get_uncertainty_df(truth, vbdf);
vb_uscore = AccuracyBenchmark.score_uncertainty(vb_udf);

mc_udf = AccuracyBenchmark.get_uncertainty_df(truth, results_df);
mc_uscore = AccuracyBenchmark.score_uncertainty(mc_udf);
mc_perdf = get_percentile_df(truth, results)

println("VB Uncertainty")
println(vb_uscore)
println("MC Uncertainty")
println(mc_uscore)

# output to output dir
CSV.write(joinpath(output_dir, "uscore_vb.csv"), vb_uscore)
CSV.write(joinpath(output_dir, "uscore_mc.csv"), mc_uscore)
CSV.write(joinpath(output_dir, "udf_vb.csv"), vb_udf)
CSV.write(joinpath(output_dir, "udf_mc.csv"), mc_udf)
CSV.write(joinpath(output_dir, "perdf_mc.csv"), mc_perdf)

########################################################################
# Compute bootstrap resampled uncertainties for prediction estimates   #
########################################################################
function score_predictions_bootstrap(truth::DataFrame,
                                     prediction_dfs::Vector{DataFrame};
                                     num_bootstrap=1000,
                                     lo=2.5,
                                     hi=97.5)
    matched_truth, matched_prediction_dfs = AccuracyBenchmark.match_catalogs(truth, prediction_dfs)

    # in each sample, resample matched truth and matched pred dataframe rows
    scoredfs = []
    for bsamp in 1:num_bootstrap
        nsamps = size(matched_truth,1)
        bidx = [rand(1:nsamps) for n in 1:nsamps]
        mtruth = matched_truth[bidx,:]
        mpreds = [m[bidx,:] for m in matched_prediction_dfs]

        error_dfs = [AccuracyBenchmark.get_error_df(mtruth, predictions) for predictions in mpreds]
        @assert length(prediction_dfs) <= 2
        sdf = AccuracyBenchmark.get_scores_df(
            matched_truth,
            error_dfs[1],
            length(error_dfs) > 1 ? Nullable(error_dfs[2]) : Nullable{DataFrame}(),
        )
        sdf[:sample_i] = ones(Int64, size(sdf,1))*bsamp
        push!(scoredfs, sdf)
    end

    # assemble hi and lo values in the bootstrap samples
    sdf = scoredfs[1]
    boot_dict = Dict(:first_lo => [],
                     :first_hi => [])
                     #:field => [])
    if :second in names(sdf)
        boot_dict[:second_lo] = []
        boot_dict[:second_hi] = []
    end

    nrows = size(sdf, 1)
    for r in 1:nrows
        first_vals  = [sdf[r,:first] for sdf in scoredfs]
        push!(boot_dict[:first_lo], percentile(first_vals, lo))
        push!(boot_dict[:first_hi], percentile(first_vals, hi))
        if :second in names(sdf)
            second_vals = [sdf[r,:second] for sdf in scoredfs]
            push!(boot_dict[:second_lo], percentile(second_vals, lo))
            push!(boot_dict[:second_hi], percentile(second_vals, hi))
        end
        #push!(boot_dict[:field], sdf[r,:field])
    end

    scoremean = AccuracyBenchmark.score_predictions(truth, prediction_dfs)
    score_cis = DataFrame(boot_dict)
    return hcat(scoremean, score_cis), vcat(scoredfs...)
end

lo, hi = 2.5, 97.5
scoredf, scoredf_boot = score_predictions_bootstrap(truth, [vbdf, results_df]);
println("Bootstrap scoredf:")
println(scoredf)
CSV.write(joinpath(output_dir, "scoredf.csv"), scoredf)
CSV.write(joinpath(output_dir, "scoredf_boot.csv"), scoredf_boot)


##############################
# look at star ROC curve     #
##############################
#matched_truth, matched_prediction_dfs = AccuracyBenchmark.match_catalogs(
#    truth, [vbdf, results_df, photodf]) #, tol=.396/3600*4)
#@printf " ... found %d matched sources for comparison \n" size(matched_truth, 1)

# look at ROC curve for pstar 
true_star = matched_truth[:is_star]
pstar_vb  = matched_prediction_dfs[1][:is_star]
pstar_mc  = matched_prediction_dfs[2][:is_star]
pstar_df = DataFrame(Dict(:true_star => true_star,
                          :pstar_vb => pstar_vb,
                          :pstar_mc => pstar_mc))
CSV.write(joinpath(output_dir, "pstardf.csv"), pstar_df)
CSV.write(joinpath(output_dir, "matched_truth.csv"), matched_truth)
CSV.write(joinpath(output_dir, "matched_vb.csv"), matched_prediction_dfs[1])
CSV.write(joinpath(output_dir, "matched_mc.csv"), matched_prediction_dfs[2])
CSV.write(joinpath(output_dir, "matched_photo.csv"), matched_prediction_dfs[3])
