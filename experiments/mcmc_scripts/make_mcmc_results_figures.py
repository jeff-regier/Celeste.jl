"""
Creates all figures given directory of AIS-MCMC result csvs
"""
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns
import pandas as pd
import numpy as np
import os, argparse
parser = argparse.ArgumentParser(description='make results figures')
parser.add_argument("--results-dir", type=str, default="s82-results", help="s82-results | s82-results-robust | mcmc_result_dataframes")
parser.add_argument("--timing-output", type=str, default=None, help="timing-output directory")
args, _ = parser.parse_known_args()
print("Making figures for results dir: ", args.results_dir)

# results dir contains CSVs and is output destination
results_dir = args.results_dir
if "s82-results" in results_dir:
    compare_to_photo = True
else:
    compare_to_photo = False

# laod in dataframes
pstardf = pd.read_csv(os.path.join(results_dir, "pstardf.csv"))
scoredf = pd.read_csv(os.path.join(results_dir, "scoredf.csv"))
scoredf_boot = pd.read_csv(os.path.join(results_dir, "scoredf_boot.csv"))

# load in matched dataframes, remove NANs
truedf = pd.read_csv(os.path.join(results_dir, "matched_truth.csv"))
vbdf = pd.read_csv(os.path.join(results_dir, "matched_vb.csv"))
mcdf = pd.read_csv(os.path.join(results_dir, "matched_mc.csv"))
photodf = pd.read_csv(os.path.join(results_dir, "matched_photo.csv"))

# remove nan obs
bad_idx = np.isnan(vbdf.flux_r_nmgy) | np.isnan(mcdf.flux_r_nmgy)
truedf = truedf[~bad_idx]
vbdf   = vbdf[~bad_idx]
mcdf   = mcdf[~bad_idx]
photodf = photodf[~bad_idx]

# set up color scheme
vb_color = sns.color_palette()[0]
mc_color = sns.color_palette()[1]

# parameter names
continuous_params = ['position', 'flux_r_nmgy', #'flux_r_mag',
                     'color_ug', 'color_gr', 'color_ri', 'color_iz',
                     'gal_frac_dev', 'gal_axis_ratio',
                     'gal_radius_px', 'gal_angle_deg']


#################
# Script Start  #
#################

def main():

    #make_star_gal_roc_curves()
    #sys.exit()
    #make_calibration_tables()
    #make_est_vs_error_plots()
    #plt.close("all")

    ## true vs predicted error plots w/ uncertainties
    #make_mcmc_vb_uncertainty_comparison_plots(source_type="star")
    #make_mcmc_vb_uncertainty_comparison_plots(source_type="gal")
    #plt.close("all")
    make_error_comparison_figs(source_type="star", error_type="abs")
    make_error_comparison_figs(source_type="gal", error_type="abs")
    plt.close("all")

    #make_error_comparison_figs(error_type="abs")
    #make_error_comparison_figs(error_type="diff")
    #make_method_comparison_figs()

    # create timing figures
    if args.timing_output is not None:
        make_timing_figures()


#####################
# Uncertainty table #
#####################

def make_calibration_tables():
    """ save calibration (within x sds) table to a latex-format table """
    def save_table(stub="uscore_mc.csv"):
        uscoredf = pd.read_csv(os.path.join(results_dir, stub))
        uscoredf.rename(columns={'field': 'parameter',
                                 'within_half_sd': "within 1/2 sd",
                                 'within_1_sd'   : "1 sd",
                                 'within_2_sd'   : "2 sd",
                                 'within_3_sd'   : "3 sd"}, inplace=True)
        uscoredf.set_index("parameter", inplace=True)
        uscoredf.rename(index={'log_flux_r_nmgy': 'log r-flux',
                               'color_ug'      : 'color ug',
                               'color_gr'      : 'color gr',
                               'color_ri'      : 'color ri',
                               'color_iz'      : 'color iz'}, inplace=True)
        print(uscoredf.head())
        formatters = [lambda x: "%2.3f"%x for _ in range(4)]
        fout = os.path.splitext(stub)[0]
        uscoredf.to_latex(os.path.join(results_dir, fout + ".tex"), formatters=formatters)

    save_table("uscore_vb.csv")
    save_table("uscore_mc.csv")


##################
# figure methods #
##################

def make_est_vs_error_plots():
    print(".... making Error Plots --- Takes a few minutes to render")
    import pyprind

    # load in matched dataframes, remove NANs
    truedf = pd.read_csv(os.path.join(results_dir, "matched_truth.csv"))
    vbdf   = pd.read_csv(os.path.join(results_dir, "matched_vb.csv"))
    mcdf   = pd.read_csv(os.path.join(results_dir, "matched_mc.csv"))
    vbdf['log_flux_r'] = np.log(vbdf.flux_r_nmgy)

    # fix gal angles
    mcdf['gal_angle_deg'][mcdf['gal_angle_deg'] < 0.] += 180.

    # only compare inferences when all truth, VB and MC agree
    star_idxs = (truedf.is_star) & (vbdf.is_star > .5) & (mcdf.is_star > .5)
    gal_idxs  = (~truedf.is_star) & (vbdf.is_star <= .5) & (mcdf.is_star <= .5)

    # remove nan obs
    bad_idx = (np.isnan(vbdf.flux_r_nmgy) | np.isnan(mcdf.flux_r_nmgy)).values

    def scatter_error(x, y, yerr, marker, label, c, alpha=.5, ax=None):
        if ax is None:
            fig, ax = plt.figure(figsize=(8,6)), plt.gca()
        ax.errorbar(x, y, yerr=yerr, ecolor=c, fmt="none")
        ax.scatter(x, y, marker=marker, label=label, c=c, s=3)
        return ax

    def plot_param_source(param_name, source_type="star"):
        #source_type = "gal"
        #param_name = "log_flux_r"
        if source_type == "star":
            idxs = star_idxs & (~bad_idx)
        elif source_type == "gal":
            idxs = gal_idxs & (~bad_idx)
        else:
            raise Exception("star|gal")

        # values to compare
        true_vals = truedf[param_name][idxs].values
        mc_means = mcdf[param_name][idxs].values
        vb_means = vbdf[param_name][idxs].values

        print("--- param %s, (source %s) ---- "%(param_name, source_type))
        print(" true vals have %d " % np.sum(pd.isnull(true_vals)))
        print(" vb preds  have %d " % np.sum(pd.isnull(vb_means)))
        print(" mc preds  have %d " % np.sum(pd.isnull(mc_means)))

        if param_name in qq_params:
            mc_errs  = 2*mcdf[param_name+"_stderr"][idxs].values
            vb_errs  = 2*vbdf[param_name+"_stderr"][idxs].values
        else:
            mc_errs = None
            vb_errs = None

        # higlight stars and gals
        fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
        lo = min(np.nanmin(true_vals), np.nanmin(mc_means), np.nanmin(vb_means))
        hi = max(np.nanmax(true_vals), np.nanmax(mc_means), np.nanmax(vb_means))
        print("lo, hi", lo, hi)
        for ax in axarr.flatten():
            ax.plot([lo, hi], [lo, hi], "--", c='grey', linewidth=2)

        scatter_error(true_vals, vb_means, vb_errs,
                        marker='o', label="VB (%s)"%source_type, c=vb_color, ax=axarr[0])
        scatter_error(true_vals, mc_means, mc_errs,
                        marker='o', label="MC (%s)"%source_type, c=mc_color, ax=axarr[1])
        axarr[0].set_xlabel("coadd value")
        axarr[1].set_xlabel("coadd value")
        axarr[0].set_ylabel("VI predicted")
        axarr[1].set_ylabel("MC predicted")
        #fig.suptitle(param_name + " (%s) "%source_type)
        fig.savefig(os.path.join(results_dir, "error-scatter-%s-%s.png"%(param_name, source_type)), bbox_inches='tight', dpi=200)

        if 'objid' in truedf.columns:
            # Print out some bad object ids for outiers
            rmses = np.abs(true_vals - mc_means)
            worst_idx = np.argsort(rmses)[::-1][:10]
            from collections import OrderedDict
            print("Objects with highest error for %s-%s params"%(source_type, param_name))
            print(pd.DataFrame(OrderedDict([
                ('objid', truedf[idxs].objid.iloc[worst_idx]),
                ('rmses', rmses[worst_idx]),
                ('true',  true_vals[worst_idx]),
                ('mc'  ,  mc_means[worst_idx]),
                ('vb'  , vb_means[worst_idx])])))

    qq_params = ["log_flux_r", "color_ug", "color_gr", "color_ri", "color_iz"]
    for source_type in ["star", "gal"]:
        for param_name in qq_params:
            plot_param_source(param_name, source_type=source_type)

    gal_params = [ 'gal_frac_dev', 'gal_axis_ratio',
                   'gal_radius_px', 'gal_angle_deg']
    for gp in gal_params:
        plot_param_source(gp, source_type="gal")


def make_qq_plots(use_percentile=False, highlight_type=False):
    """ QQ Plot for Color and Log Ref Band Flux """
    from scipy import stats
    from statsmodels.graphics.gofplots import qqplot
    from scipy.stats import norm

    # load error data frames, and percentile dataframes
    perdf_mc = pd.read_csv(os.path.join(results_dir, "perdf_mc.csv"))
    udf_mc   = pd.read_csv(os.path.join(results_dir, "udf_mc.csv"))
    udf_vb   = pd.read_csv(os.path.join(results_dir, "udf_vb.csv"))

    # load in matched dataframes, remove NANs
    truedf = pd.read_csv(os.path.join(results_dir, "matched_truth.csv"))
    vbdf = pd.read_csv(os.path.join(results_dir, "matched_vb.csv"))
    mcdf = pd.read_csv(os.path.join(results_dir, "matched_mc.csv"))

    # remove nan obs
    bad_idx = (np.isnan(vbdf.flux_r_nmgy) | np.isnan(mcdf.flux_r_nmgy)).values
    is_star = truedf.is_star[~bad_idx].values

    def plot_param(param_name):
        # select errors for MC and VB
        udf_mc_param = udf_mc[udf_mc.name==param_name][~bad_idx]
        udf_vb_param = udf_vb[udf_vb.name==param_name][~bad_idx]
        if param_name == "log_flux_r_nmgy":
            perdf_mc_param = perdf_mc[perdf_mc.name=="log_flux_r"][~bad_idx]
        else:
            perdf_mc_param = perdf_mc[perdf_mc.name==param_name][~bad_idx]

        # parameter MC and VB z scores
        zs_vb = udf_vb_param.error / udf_vb_param.posterior_std_err
        zs_mc = udf_mc_param.error / udf_mc_param.posterior_std_err
        if use_percentile:
            per_vals = np.clip(perdf_mc_param['percentile'].values, 1./1000, 1-1./1000)
            zs_mc = norm.ppf(per_vals)

        # compute qq plot scores for MC and VB
        (osm, osr), _ = stats.probplot(zs_mc)
        (osm_vb, osr_vb), _ = stats.probplot(zs_vb)

        # ylo/hi based on percentiles
        figsize = (8,6)
        fig, ax = plt.figure(figsize=figsize), plt.gca()
        ylo, yhi = np.nanpercentile(np.concatenate([osr, osr_vb]), [1, 99])

        # plot, subtly marking Stars vs Galaxies
        ax.scatter(osm_vb[is_star], osr_vb[is_star], label="VI (star)", marker='x', alpha=1., c=vb_color)
        ax.scatter(osm_vb[~is_star], osr_vb[~is_star], label="VI (gal)", marker='x', alpha=.5, c=vb_color)
        ax.scatter(osm[is_star], osr[is_star], label="MC (star)", marker='o', alpha=1., c=mc_color)
        ax.scatter(osm[~is_star], osr[~is_star], label="MC (gal)", marker='o', alpha=.5, c=mc_color)
        lo, hi = max(ax.get_xlim()[0], ax.get_ylim()[0]), min(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "--", linewidth=3, c='grey', alpha=.9)
        ax.legend(fontsize=14)
        ax.set_ylim(ylo, yhi)
        #ax.set_xlim(ylo, yhi)
        ax.set_xlabel("theoretical")
        ax.set_ylabel("observed")
        fig.savefig(os.path.join(results_dir, "qq-%s.png"%param_name), bbox_inches='tight', dpi=200)

        # plot separately star and galaxies
        fig, ax = plt.figure(figsize=figsize), plt.gca()
        ax.scatter(osm_vb[is_star], osr_vb[is_star], label="VI (star)", marker='x', alpha=1.)
        ax.scatter(osm[is_star], osr[is_star], label="MC (star)", marker='o', alpha=1.)
        lo, hi = max(ax.get_xlim()[0], ax.get_ylim()[0]), min(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "--", linewidth=3, c='grey', alpha=.9)
        ax.legend(fontsize=14)
        ax.set_ylim(ylo, yhi)
        #ax.set_xlim(ylo, yhi)
        ax.set_xlabel("theoretical")
        ax.set_ylabel("observed")
        fig.savefig(os.path.join(results_dir, "qq-%s-star.png"%param_name), bbox_inches='tight', dpi=200)

        # plot separately star and galaxies
        fig, ax = plt.figure(figsize=figsize), plt.gca()
        ax.scatter(osm_vb[~is_star], osr_vb[~is_star], label="VI (gal)", marker='x', alpha=1.)
        ax.scatter(osm[~is_star], osr[~is_star], label="MC (gal)", marker='o', alpha=1.)
        lo, hi = max(ax.get_xlim()[0], ax.get_ylim()[0]), min(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "--", linewidth=3, c='grey', alpha=.9)
        ax.legend(fontsize=14)
        ax.set_ylim(ylo, yhi)
        #ax.set_xlim(ylo, yhi)
        ax.set_xlabel("theoretical")
        ax.set_ylabel("observed")
        fig.savefig(os.path.join(results_dir, "qq-%s-gal.png"%param_name), bbox_inches='tight', dpi=200)

    # for each continuous parameter type
    qq_params = ["log_flux_r_nmgy", "color_ug", "color_gr", "color_ri", "color_iz"]
    for param_name in qq_params:
        plot_param(param_name)


def make_error_comparison_figs(source_type="star", error_type="abs"):
    """ Make Violin Plot Error Comparison figures """

    # construct parameter data frame 
    def param_df(param_name, truedf, vbdf, mcdf, photodf=None, error_type="abs"):
        """ create error df for different methods """
        if param_name == 'position':
            true_pos = np.column_stack([ truedf.ra.values, truedf.dec.values ])
            def dist(df):
                pos = np.column_stack([ df.ra.values, df.dec.values ])
                return np.sqrt(np.sum((true_pos - pos)**2, axis=1))

            dvb, dmc = dist(vbdf), dist(mcdf)
            method_list = np.concatenate([['vb']*len(dvb), ['mc']*len(dmc)])
            error_list  = np.concatenate([ dvb, dmc ])
            if photodf is not None:
                dphoto = dist(photodf)
                method_list = np.concatenate([method_list, ['photo']*len(dphoto)])
                error_list  = np.concatenate([error_list, dphoto])

        elif param_name=="flux_r_nmgy":
            dmc = np.log(truedf[param_name].values) - np.log(mcdf[param_name].values)
            dvb = np.log(truedf[param_name].values) - np.log(vbdf[param_name].values)
            method_list = np.concatenate([['vb']*len(dvb), ['mc']*len(dmc)])
            error_list  = np.concatenate([ dvb, dmc ])
            if photodf is not None:
                dphoto = np.log(truedf[param_name].values) - np.log(photodf[param_name].values)
                method_list = np.concatenate([method_list, ['photo']*len(dphoto)])
                error_list  = np.concatenate([error_list, dphoto])

        elif param_name=="gal_angle_deg":
            def angle_error(true, pred):
                diffs = np.column_stack([true - pred,
                                         true-(pred-180.),
                                         true-(pred+180.)])
                mini  = np.argmin(np.abs(diffs), axis=1)
                return diffs[np.arange(len(mini)), mini]

            dmc = angle_error(truedf[param_name].values, mcdf[param_name].values)
            dvb = angle_error(truedf[param_name].values, vbdf[param_name].values)
            method_list = np.concatenate([['vb']*len(dvb), ['mc']*len(dmc)])
            error_list  = np.concatenate([ dvb, dmc ])
            if photodf is not None:
                dphoto = angle_error(truedf[param_name].values, photodf[param_name].values)
                method_list = np.concatenate([method_list, ['photo']*len(dphoto)])
                error_list  = np.concatenate([error_list, dphoto])
        else:
            dmc = truedf[param_name].values - mcdf[param_name].values
            dvb = truedf[param_name].values - vbdf[param_name].values
            method_list = np.concatenate([['vb']*len(dvb), ['mc']*len(dmc)])
            error_list  = np.concatenate([ dvb, dmc ])
            if photodf is not None:
                dphoto = truedf[param_name].values - photodf[param_name].values
                method_list = np.concatenate([method_list, ['photo']*len(dphoto)])
                error_list  = np.concatenate([error_list, dphoto])

        if error_type=="abs":
            error_list = np.abs(error_list)

        # consistent naming
        outdf = pd.DataFrame({'method': method_list, 'error':error_list})
        outdf.method[outdf.method=='mc'] = 'MC'
        outdf.method[outdf.method=='vb'] = 'VI'
        return outdf

    ##################
    # start function #
    ##################
    # load in matched dataframes, remove NANs
    truedf = pd.read_csv(os.path.join(results_dir, "matched_truth.csv"))
    vbdf   = pd.read_csv(os.path.join(results_dir, "matched_vb.csv"))
    mcdf   = pd.read_csv(os.path.join(results_dir, "matched_mc.csv"))
    vbdf['log_flux_r'] = np.log(vbdf.flux_r_nmgy)
    mcdf['gal_angle_deg'][mcdf['gal_angle_deg'] < 0.] += 180.  # fix gal angles

    # only compare inferences when all truth, VB and MC agree
    star_idxs = (truedf.is_star) & (vbdf.is_star > .5) & (mcdf.is_star > .5)
    gal_idxs  = (~truedf.is_star) & (vbdf.is_star <= .5) & (mcdf.is_star <= .5)
    # remove nan obs
    bad_idx = (np.isnan(vbdf.flux_r_nmgy) | np.isnan(mcdf.flux_r_nmgy)).values

    # match up stars to gals
    truedf = pd.concat([ truedf[star_idxs & (~bad_idx)], truedf[gal_idxs &(~bad_idx)] ])
    vbdf = pd.concat([ vbdf[star_idxs & (~bad_idx)], vbdf[gal_idxs &(~bad_idx)] ])
    mcdf = pd.concat([ mcdf[star_idxs & (~bad_idx)], mcdf[gal_idxs &(~bad_idx)] ])

    if compare_to_photo:
        photodf = pd.read_csv(os.path.join(results_dir, "matched_photo.csv"))
        photodf = pd.concat([ photodf[star_idxs & (~bad_idx)], photodf[gal_idxs &(~bad_idx)] ])

    fig, axarr = plt.subplots(2, int(len(continuous_params)/2), figsize=(12, 6))
    for ax, cp in zip(axarr.flatten(), continuous_params):
        if compare_to_photo:
            pdf = param_df(cp, truedf, vbdf, mcdf, photodf=photodf, error_type=error_type)
        else:
            pdf = param_df(cp, truedf, vbdf, mcdf, photodf=None, error_type=error_type)
        print(pdf.min())
        sns.violinplot(x="method", y="error", data=pdf, ax=ax, bw=.2) #'scott')
        pretty_labels = {'flux_r_nmgy': "r-band flux",
                         'color_ug'   : "color ug",
                         'color_gr'   : "color gr",
                         'color_ri'   : "color ri",
                         'color_iz'   : "color iz",
                         'gal_frac_dev': "de vaucouleurs frac (gal)",
                         'gal_axis_ratio': "axis ratio (gal)",
                         'gal_radius_px' : "radius in px (gal)",
                         'gal_angle_deg' : "angle in degrees (gal)",
                         'position'      : "position" }
        ax.set_xlabel(pretty_labels[cp])

        if cp is not "position":
            ylo, yhi = ax.get_ylim()
            if error_type=="abs":
                ylim = np.nanpercentile(pdf['error'], [0, 97.5])
            elif error_type=="diff":
                ylim = np.nanpercentile(pdf['error'], [2.5, 97.5])
            ax.set_ylim(ylim)

    fig.tight_layout()
    fname = os.path.join(results_dir, "error_vb_mc_comparison_%s-%s.png"%(source_type, error_type))
    fig.savefig(fname, bbox_inches='tight', dpi=200)

#################
# ROC Curves    #
#################

def make_star_gal_roc_curves():
    """ Star/Gal ROC curves """
    import pyprind
    from sklearn.metrics import roc_curve, roc_auc_score
    N = len(pstardf)
    Y   = pstardf['true_star'].values
    pmc = pstardf['pstar_mc'].values
    pvb = pstardf['pstar_vb'].values

    vb_col = sns.color_palette()[0]
    mc_col = sns.color_palette()[1]

    fig, ax = plt.figure(figsize=(6,4)), plt.gca()

    #mc_boot, vb_boot = [], []
    #for i in pyprind.prog_bar(range(50)):
    #    bids = np.random.choice(N, size=N)
    #    vb_fpr, vb_tpr, thresh = metrics.roc_curve(Y[bids], pvb[bids], drop_intermediate=False)
    #    mc_fpr, mc_tpr, thresh = metrics.roc_curve(Y[bids], pmc[bids], drop_intermediate=False)
    #    ax.plot(vb_fpr, vb_tpr, c=vb_col, alpha=.1)
    #    ax.plot(mc_fpr, mc_tpr, c=mc_col, alpha=.1)
    #    #mc_boot.append(roc_curve(Y[bids], pmc[bids]))
    #    #vb_boot.append(roc_curve(Y[bids], pvb[bids]))

    mc_fpr, mc_tpr, thresh = roc_curve(Y, pmc)
    vb_fpr, vb_tpr, thresh = roc_curve(Y, pvb)
    ax.plot(mc_fpr, mc_tpr, label="MC", linewidth=3)
    ax.plot(vb_fpr, vb_tpr, label="VI", linewidth=3)
    if compare_to_photo:
        photo_fpr = np.sum((photodf.is_star) & (~truedf.is_star)) / np.sum(~truedf.is_star)
        photo_tpr = np.sum((photodf.is_star) & (truedf.is_star)) / np.sum(truedf.is_star)
        ax.scatter(photo_fpr, photo_tpr, marker="x", color='red', linewidth=2, s=35, label="Photo")

    ax.legend(fontsize=14)
    ax.set_xlim(0, .4)
    ax.set_ylim(.6, 1.)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig.savefig(os.path.join(results_dir, "pstar_roc_comparison.png"), bbox_inches='tight', dpi=200)

    # compute AUC for MC and VB
    def make_mc_vb_auc_df():
        mc_aucs, vb_aucs = [], []
        for i in range(5000):
            idx = np.random.choice(len(Y), size=len(Y))
            mc_aucs.append(roc_auc_score(Y[idx], pmc[idx]))
            vb_aucs.append(roc_auc_score(Y[idx], pvb[idx]))
        return pd.DataFrame({'inference method': ['MC']*len(mc_aucs) + ['VI']*len(vb_aucs),
                             'AUC'   : np.concatenate([mc_aucs, vb_aucs])})

    aucdf = make_mc_vb_auc_df()
    mc_auc = roc_auc_score(Y, pmc)
    vb_auc = roc_auc_score(Y, pvb)
    mc_aucs = aucdf['AUC'][aucdf['inference method'] == 'MC']
    vb_aucs = aucdf['AUC'][aucdf['inference method'] == 'VI']
    print("======= Star vs. Galaxy AUC Scores ==========")
    print("   MC : %2.4f  [%2.4f,  %2.4f] "%(mc_auc, np.percentile(mc_aucs, 2.5), np.percentile(mc_aucs, 97.5)))
    print("   VB : %2.4f  [%2.4f,  %2.4f] "%(vb_auc, np.percentile(vb_aucs, 2.5), np.percentile(vb_aucs, 97.5)))

    fig, ax = plt.figure(figsize=(6,4)), plt.gca()
    sns.violinplot(x='inference method', y='AUC', data=aucdf)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "pstar_auc_comparison.png"), bbox_inches='tight', dpi=200)


def make_mcmc_vb_uncertainty_comparison_plots(source_type="star", param_name="log_flux_r"):
    """ Compare MCMC and VB posterior uncertainty on examples where the
    prediction is way off.
      - These plots show that MCMC have better uncertainty properties
    """
    # load in matched dataframes, remove NANs
    truedf = pd.read_csv(os.path.join(results_dir, "matched_truth.csv"))
    vbdf   = pd.read_csv(os.path.join(results_dir, "matched_vb.csv"))
    mcdf   = pd.read_csv(os.path.join(results_dir, "matched_mc.csv"))
    vbdf['log_flux_r'] = np.log(vbdf.flux_r_nmgy)
    mcdf['gal_angle_deg'][mcdf['gal_angle_deg'] < 0.] += 180.

    # only compare inferences when all truth, VB and MC agree
    star_idxs = (truedf.is_star) & (vbdf.is_star > .5) & (mcdf.is_star > .5)
    gal_idxs  = (~truedf.is_star) & (vbdf.is_star <= .5) & (mcdf.is_star <= .5)
    # remove nan obs
    bad_idx = (np.isnan(vbdf.flux_r_nmgy) | np.isnan(mcdf.flux_r_nmgy)).values

    if source_type == "star":
        idxs = star_idxs & (~bad_idx)
    elif source_type == "gal":
        idxs = gal_idxs & (~bad_idx)
    else:
        raise Exception("sourcetype = star|gal")

    # find big errors
    true_rmag = truedf[idxs][param_name].values
    vb_rmag   = vbdf[idxs][param_name].values
    mc_rmag   = mcdf[idxs][param_name].values
    vb_stderr = vbdf[idxs][param_name+"_stderr"].values
    mc_stderr = mcdf[idxs][param_name+"_stderr"].values

    # find locations where VB is way off
    vberr = np.abs(true_rmag - vb_rmag)
    mcerr = np.abs(true_rmag - mc_rmag)
    bad_idx = np.argsort(vberr)[::-1]

    for idx in bad_idx[:10]:
        print("two errs: ", vberr[idx], mcerr[idx])
        vbmu, vbscale = vb_rmag[idx], vb_stderr[idx]
        mcmu, mcscale = mc_rmag[idx], mc_stderr[idx]
        true_val = true_rmag[idx]
        print(vbmu, vbscale)
        print(mcmu, mcscale)

        from scipy.stats import norm
        cvb, cmc, ctrue = sns.color_palette()[0], sns.color_palette()[1], sns.color_palette()[3]
        fig, ax = plt.figure(figsize=(8,4)), plt.gca()
        lo, hi = min(mcmu - 2.75*mcscale, vbmu-2.75*vbscale, true_val), \
                 max(mcmu + 2.75*mcscale, vbmu+2.75*vbscale, true_val)
        xgrid = np.linspace(lo, hi, 1000)
        ax.plot(xgrid, norm.pdf(xgrid, vbmu, vbscale), label="VI", linewidth=2, c=cvb)
        ax.fill_between(xgrid, norm.pdf(xgrid, vbmu, vbscale), alpha=.5, color=cvb)
        ax.plot(xgrid, norm.pdf(xgrid, mcmu, mcscale), "--", label="MC", linewidth=2, c=cmc)
        ax.fill_between(xgrid, norm.pdf(xgrid, mcmu, mcscale), alpha=.5, color=cmc)
        ax.scatter(true_val, 0., s=200, linewidth=3, marker='x', label="true", c=ctrue)
        ax.set_xlabel("log r-band flux")
        ax.legend(fontsize=14)
        fig.savefig(os.path.join(results_dir, "posterior-comparison-%s-%s-src-%d.png"%(source_type, param_name, idx)), bbox_inches='tight')


def make_timing_figures():
    timedf = pd.read_csv('timing-output/timedf.csv')
    # print time
    vb_time = timedf[timedf.method=='vb'].time.iloc[0]
    mc_timedf = timedf[timedf.method=='mc']
    #mc_time   = mc_timedf.time #np.concatenate([[.01], mc_timedf.time])
    #ess_star  = mc_timedf.ess_star #np.concatenate([[.01], mc_timedf.ess_star])
    #ess_gal   = mc_timedf.ess_gal #np.concatenate([[.01], mc_timedf.ess_gal])
    #mc_nsamps = mc_timedf.nsamps #np.concatenate([[.01], mc_timedf.nsamps])
    mc_time   = np.concatenate([[0], mc_timedf.time])
    ess_star  = np.concatenate([[0], mc_timedf.ess_star])
    ess_gal   = np.concatenate([[0], mc_timedf.ess_gal])
    mc_nsamps = np.concatenate([[0], mc_timedf.nsamps])

    # MCMC moves vs time
    fig, ax = plt.figure(figsize=(8,4)), plt.gca()
    ax.plot(mc_time, mc_nsamps, label="mcmc-transitions", c=mc_color)
    ylim = ax.get_ylim()
    ax.plot([vb_time, vb_time], ylim, label="VI", c=vb_color)
    ax.set_ylim(ylim)
    ax.legend()
    ax.set_xlabel("Time (seconds)", fontsize=14)
    ax.set_ylabel("Number of MCMC Transitions")
    fig.savefig('timing-output/time-nsamps-fig.png', bbox_inches='tight', dpi=200)

    # Number of Effecive Samples
    fig, ax = plt.figure(figsize=(8,4)), plt.gca()
    ax.plot(mc_time, ess_star, "--o", label="star ess", c=mc_color)
    ax.plot(mc_time, ess_gal, "-->", label="gal ess", c=mc_color)
    ylim = ax.get_ylim()
    ax.plot([vb_time, vb_time], ylim, label="VI", linewidth=2)
    ax.set_ylim(ylim)
    ax.legend()
    ax.set_xlabel("Time (seconds)", fontsize=14)
    ax.set_ylabel("Eff. Ind. Samples", fontsize=14)
    fig.tight_layout()
    fig.savefig('timing-output/time-ess-fig.png', bbox_inches='tight', dpi=200)



##############
# run script #
##############
if __name__=="__main__":
    main()


############
# ded code #
############

########################################
# create-per-parameter violin plots
########################################
#def make_violin_plots():
#    boot_dict = {
#        "method": ["vb"]*len(scoredf_boot) + ["mc"]*len(scoredf_boot),
#        "error": np.concatenate([scoredf_boot['first'].values,
#                               scoredf_boot['second'].values]),
#        "field": np.concatenate([scoredf_boot.field.values, 
#                                 scoredf_boot.field.values])
#        }
#
#    bdf = pd.DataFrame(boot_dict)
#
#    fields = bdf.field.unique()
#    fields = fields[ fields != 'flux_r_nmgy' ]
#    fig, axarr = plt.subplots(2, len(fields)/2, figsize=(12, 8))
#    for ax, field in zip(axarr.flatten(), fields):
#        sns.violinplot(x="field", y="error", hue="method",
#                       data=bdf[bdf.field==field], split=True, ax=ax)
#        #ax.set_title("%s"%field)
#        ax.set_xlabel("")
#
#        if field != fields[0]:
#            ax.legend_.remove()
#
#    #fig.tight_layout()
#    #fig.savefig(os.path.join(results_dir, "error_vb_mc_comparison.png"), bbox_inches='tight', dpi=200)
#
#    # create scoredf latex table
#    #scoredf.rename({"first": "vb", "second": "mc"})
#    def write_interval(scores, lo, hi):
#        return ["%2.2f [%2.2f, %2.2f]"%(s, l, h) for s, l, h in zip(scores, lo, hi)]
#
#    tabdict = {
#        "N" : scoredf.N,
#        "vb": write_interval(scoredf['first'], scoredf['first_lo'], scoredf['first_hi']),
#        "mc": write_interval(scoredf['second'], scoredf['second_lo'], scoredf['second_hi']),
#        "field": scoredf.field
#    }
#
#    tabdf = pd.DataFrame(tabdict)
#    with open(os.path.join(results_dir, "error_table.tex"), 'w') as f:
#        f.write(tabdf.to_latex())



##########################################################
# Method-Method comparison for each source figure making #
##########################################################
#def source_error_df(param_name, truedf, vbdf, mcdf, photodf=None):
#    """ create error df for different methods """
#    if param_name == 'position':
#        true_pos = np.column_stack([ truedf.ra.values, truedf.dec.values ])
#        def dist(df):
#            pos = np.column_stack([ df.ra.values, df.dec.values ])
#            return np.sqrt(np.sum((true_pos - pos)**2, axis=1))
#
#        dvb, dmc = dist(vbdf), dist(mcdf)
#        method_list = np.concatenate([['vb']*len(dvb), ['mc']*len(dmc)])
#        error_list  = np.concatenate([ dvb, dmc ])
#        if photodf is not None:
#            dphoto = dist(photodf)
#            method_list = np.concatenate([method_list, ['photo']*len(dphoto)])
#            error_list  = np.concatenate([error_list, dphoto])
#
#    elif param_name=="flux_r_nmgy":
#        dmc = np.log(truedf[param_name].values) - np.log(mcdf[param_name].values)
#        dvb = np.log(truedf[param_name].values) - np.log(vbdf[param_name].values)
#        method_list = np.concatenate([['vb']*len(dvb), ['mc']*len(dmc)])
#        error_list  = np.concatenate([ dvb, dmc ])
#        if photodf is not None:
#            dphoto = np.log(truedf[param_name].values) - np.log(photodf[param_name].values)
#            method_list = np.concatenate([method_list, ['photo']*len(dphoto)])
#            error_list  = np.concatenate([error_list, dphoto])
#
#    elif param_name=="gal_angle_deg":
#        def angle_error(true, pred):
#            diffs = np.column_stack([true - pred,
#                                     true-(pred-180.),
#                                     true-(pred+180.)])
#            mini  = np.argmin(np.abs(diffs), axis=1)
#            return diffs[np.arange(len(mini)), mini]
#
#        dmc = angle_error(truedf[param_name].values, mcdf[param_name].values)
#        dvb = angle_error(truedf[param_name].values, vbdf[param_name].values)
#        method_list = np.concatenate([['vb']*len(dvb), ['mc']*len(dmc)])
#        error_list  = np.concatenate([ dvb, dmc ])
#        if photodf is not None:
#            dphoto = angle_error(truedf[param_name].values, photodf[param_name].values)
#            method_list = np.concatenate([method_list, ['photo']*len(dphoto)])
#            error_list  = np.concatenate([error_list, dphoto])
#    else:
#        dmc = truedf[param_name].values - mcdf[param_name].values
#        dvb = truedf[param_name].values - vbdf[param_name].values
#        method_list = np.concatenate([['vb']*len(dvb), ['mc']*len(dmc)])
#        error_list  = np.concatenate([ dvb, dmc ])
#        if photodf is not None:
#            dphoto = truedf[param_name].values - photodf[param_name].values
#            method_list = np.concatenate([method_list, ['photo']*len(dphoto)])
#            error_list  = np.concatenate([error_list, dphoto])
#
#    error_list = np.abs(error_list)
#
#    # now compare vb to mcmc and vb to photo if available
#    n_sources = truedf.shape[0]
#    vb_err, mc_err = error_list[:n_sources], error_list[n_sources:2*n_sources]
#    if photodf is not None:
#        photo_err = error_list[2*n_sources:]
#        vb_to_photo = (photo_err-vb_err) #/ (photo_err + 1e-6)
#        mc_to_photo = (photo_err-mc_err) # / (photo_err + 1e-6)
#        #vb_to_mc    = vb_err / (mc_err + 1e-6)
#        comparison = np.concatenate([ ["vb_to_photo"]*len(vb_to_photo),
#                                      ["mc_to_photo"]*len(mc_to_photo) ])
#                                      #["vb_to_mc"]*len(vb_to_mc) ])
#        elist = np.concatenate([ vb_to_photo, mc_to_photo])
#        edf = pd.DataFrame({'comparison': comparison, 'error': elist})
#    else:
#        vb_to_mc = vb_err-mc_err
#        comparison = np.concatenate([ ["vb_to_mc"]*len(vb_to_mc) ])
#        edf = pd.DataFrame({'comparison': comparison, 'error': vb_to_mc})
#
#    # remove NaNs
#    edf = edf.dropna()
#    return edf
#
#
#def make_method_comparison_figs():
#    print "--- making method comparison --- "
#    fig, axarr = plt.subplots(2, len(continuous_params)/2, figsize=(12, 6))
#    for ax, cp in zip(axarr.flatten(), continuous_params):
#        if compare_to_photo:
#            pdf = source_error_df(cp, truedf, vbdf, mcdf, photodf=photodf)
#        else:
#            pdf = source_error_df(cp, truedf, vbdf, mcdf, photodf=None)
#        print pdf.min()
#        sns.violinplot(x="comparison", y="error", data=pdf, ax=ax, bw=.2) #'scott')
#        ax.set_xlabel(cp)
#        if cp is not "position":
#           #ylo, yhi = ax.get_ylim()
#           ylim = np.nanpercentile(pdf['error'], [2.5, 97.5])
#           ax.set_ylim(ylim)
#           #ax.set_ylim(ylo, ylim[1])
#
#    fig.tight_layout()
#    fname = os.path.join(results_dir, "source_method_comparison.png")
#    fig.savefig(fname, bbox_inches='tight', dpi=200)


