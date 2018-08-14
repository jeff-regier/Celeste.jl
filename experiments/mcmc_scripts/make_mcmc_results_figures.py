"""
Creates all figures given directory of AIS-MCMC result csvs
"""
import matplotlib
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns
import pandas as pd
import numpy as np
import os, argparse, pyprind

# set global default font sizes
fontsize = 16
matplotlib.rcParams['xtick.labelsize'] = fontsize-3
matplotlib.rcParams['ytick.labelsize'] = fontsize-3
matplotlib.rcParams['font.size']       = fontsize

# set up color scheme
vb_color = sns.color_palette()[0]
mc_color = sns.color_palette()[1]

# parameter names
continuous_params = ['position', 'flux_r_nmgy', #'flux_r_mag',
                     'color_ug', 'color_gr', 'color_ri', 'color_iz',
                     'gal_frac_dev', 'gal_axis_ratio',
                     'gal_radius_px', 'gal_angle_deg']


#####################
# Uncertainty table #
#####################

def make_calibration_tables(results_dir):
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

def make_est_vs_error_plots(results_dir):
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

        # wrap gal angles
        if param_name == 'gal_angle_deg':
            mc_wrapped = np.column_stack([ mc_means, mc_means-180, mc_means+180 ])
            mc_dist    = np.abs(mc_wrapped-true_vals[:,None])
            mc_idx     = np.argmin(mc_dist, axis=1)
            mc_means = np.array([ mc[i] for i,mc in zip(mc_idx, mc_wrapped) ])

            vb_wrapped = np.column_stack([ vb_means, vb_means-180, vb_means+180])
            vb_dist    = np.abs(vb_wrapped-true_vals[:,None])
            vb_idx     = np.argmin(vb_dist, axis=1)
            vb_means = np.array([ vb[i] for i,vb in zip(vb_idx, vb_wrapped) ])


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
        fig, axarr = plt.subplots(1, 2, figsize=(9, 3.75))
        lo = min(np.nanmin(true_vals), np.nanmin(mc_means), np.nanmin(vb_means))
        hi = max(np.nanmax(true_vals), np.nanmax(mc_means), np.nanmax(vb_means))
        print("lo, hi", lo, hi)
        for ax in axarr.flatten():
            ax.plot([lo, hi], [lo, hi], "--", c='grey', linewidth=2)

        scatter_error(true_vals, vb_means, vb_errs,
                        marker='o', label="VB (%s)"%source_type, c=vb_color, ax=axarr[0])
        scatter_error(true_vals, mc_means, mc_errs,
                        marker='o', label="MCMC (%s)"%source_type, c=mc_color, ax=axarr[1])
        if False: #compare_to_photo:
            axarr[0].set_xlabel("coadd value", fontsize=fontsize)
            axarr[1].set_xlabel("coadd value", fontsize=fontsize)
        else:
            axarr[0].set_xlabel("ground truth", fontsize=fontsize)
            axarr[1].set_xlabel("ground truth", fontsize=fontsize)
        axarr[0].set_ylabel("VI predicted", fontsize=fontsize)
        axarr[1].set_ylabel("MCMC predicted", fontsize=fontsize)
        #axarr[0].tick_params(labelsize=args.fontsize-2)
        #axarr[1].tick_params(labelsize=args.fontsize-2)
        #fig.suptitle(param_name + " (%s) "%source_type)
        fig.tight_layout()
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


def make_error_comparison_figs(results_dir, source_type="star",
                               error_type="abs", compare_to_photo=False):
    """ Make Violin Plot Error Comparison figures """

    # construct parameter data frame 
    def param_df(param_name, truedf, vbdf, mcdf, photodf=None, error_type="abs"):
        """ create error df for different methods """
        if param_name == 'position':
            true_pos = np.column_stack([ truedf.ra.values, truedf.dec.values ])
            def dist(df):
                pos = np.column_stack([ df.ra.values, df.dec.values ])
                pixel_error =  position_error(truedf.ra.values, truedf.dec.values,
                                              df.ra.values, df.dec.values)
                return pixel_error
                # return np.sqrt(np.sum((true_pos - pos)**2, axis=1))*3600.

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
                #pred_wrapped = np.column_stack([ pred, pred-180, pred+180])
                #diffs = np.abs(pred_wrapped-true[:,None])
                #min_idx = np.argmin(diffs, axis=1)
                #pred_fixed = np.array([ mc[i] for i,mc in zip(min_idx, pred_wrapped)])
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
        outdf.method[outdf.method=='mc'] = 'MCMC'
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

    pretty_labels = {'flux_r_nmgy'   : "brightness",
                     'color_ug'      : "color u-g",
                     'color_gr'      : "color g-r",
                     'color_ri'      : "color r-i",
                     'color_iz'      : "color i-z",
                     'gal_frac_dev'  : "profile", #'gal_frac_dev': "de vaucouleurs frac (gal)",
                     'gal_axis_ratio': "axis",
                     'gal_radius_px' : "radius",
                     'gal_angle_deg' : "angle",
                     'position'      : "position" }

    fig, axarr = plt.subplots(2, int(len(continuous_params)/2), figsize=(12, 6))
    for ax, cp in zip(axarr.flatten(), continuous_params):
        if compare_to_photo:
            pdf = param_df(cp, truedf, vbdf, mcdf, photodf=photodf, error_type=error_type)
        else:
            pdf = param_df(cp, truedf, vbdf, mcdf, photodf=None, error_type=error_type)
        print(pdf.min())
        vp = sns.violinplot(x="method", y="error", data=pdf, ax=ax, bw=.2) #'scott')
        ax.set_xlabel(pretty_labels[cp])
        vp.tick_params(labelsize=11)

        # make sure we use sci notation for small numbers
        if cp == "position":
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

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

    ############################################################
    # also save error table (with significance highlighting?)  #
    ############################################################
    edfs, zdfs = [], []
    sample_sizes = []
    for cp in continuous_params:
        if compare_to_photo:
            pdf = param_df(cp, truedf, vbdf, mcdf, photodf=photodf, error_type=error_type)
        else:
            pdf = param_df(cp, truedf, vbdf, mcdf, photodf=None, error_type=error_type)
        pdf['param'] = [pretty_labels[cp]]*len(pdf)
        edfs.append(pdf)

        # compute z score errors
        zdf = pdf.copy()
        zdf['error'] /= zdf.error.std()
        zdfs.append(zdf)
        sample_sizes.append(np.sum(~zdf.error.isnull()))

    edf = pd.concat(edfs, 0)
    #zdf = pd.concat(zdfs, 0)
    meandf  = edf.groupby(["param", "method"], sort=False).mean()
    countdf = edf.groupby(["param", "method"], sort=False).count()
    meandf['error stdev'] = edf.groupby(["param", "method"]).std() / np.sqrt(countdf)
    meandf.reset_index(inplace=True)
    print(meandf)

    # construct tex table
    methods = ["MCMC", "VI"]
    if compare_to_photo:
        methods += ["photo"]

    def create_pairs(zdf):
        dfs = {m: zdf[zdf.method==m].reset_index(drop=True) for m in methods}
        names = []
        df_list = []
        if compare_to_photo:
            pv_delta = dfs["photo"].copy()
            pv_delta.error -= dfs["VI"].error
            pm_delta = dfs["photo"].copy()
            pm_delta.error -= dfs["MCMC"].error
            names += ["photo-VI", "photo-MCMC"]
            df_list += [pv_delta, pm_delta]

        vm_delta = dfs["VI"].copy()
        vm_delta.error -= dfs["MCMC"].error
        names += ["VI-MCMC"]
        df_list += [vm_delta]

        str_cols = []
        for ddf, name in zip(df_list, names):
            emean = ddf.groupby(["param"], sort=False).mean().error
            ecnt  = ddf.groupby(["param"], sort=False).count().error
            estd  = ddf.groupby(["param"], sort=False).std().error / np.sqrt(ecnt)
            str_cols.append(["%2.4f ($\pm$ %2.3f)"%(e, s)
                            for e, s in zip(emean, estd)])

        from collections import OrderedDict
        odf = pd.DataFrame(OrderedDict(zip(names, str_cols)), index=emean.index)
        return odf

    str_cols = []
    for m in methods:
        mdf = meandf[meandf.method==m]
        #mstr = ["%2.3f $\pm$ %2.2f" % (e,s)
        #        for e,s in zip(mdf.error, mdf['error stdev'])]
        mstr = ["%2.3f" % e for e,s in zip(mdf.error, mdf['error stdev'])]
        str_cols.append(mstr)

    outdf  = pd.DataFrame({m: s for m, s in zip(methods, str_cols)}, index=mdf.param)
    pairdf = create_pairs(edf)
    totaldf = pd.concat([outdf, pairdf], axis=1)
    print("Totaldf: ", totaldf)

    # fout = os.path.splitext(stub)[0]
    outdf.to_latex(os.path.join(results_dir, "error_vb_mc_comparison.tex"), escape=False)
    totaldf.to_latex(os.path.join(results_dir, "error_vb_mc_comparison-pair.tex"), escape=False)


#################
# ROC Curves    #
#################

def make_star_gal_roc_curves(results_dir, compare_to_photo=False):
    """ Star/Gal ROC curves """
    from sklearn.metrics import roc_curve, roc_auc_score
    pstardf = pd.read_csv(os.path.join(results_dir, "pstardf.csv"))
    N = len(pstardf)
    Y   = pstardf['true_star'].values
    pmc = pstardf['pstar_mc'].values
    pvb = pstardf['pstar_vb'].values

    vb_col = sns.color_palette()[0]
    mc_col = sns.color_palette()[1]

    fig, ax = plt.figure(figsize=(6,4)), plt.gca()
    mc_fpr, mc_tpr, thresh = roc_curve(Y, pmc)
    vb_fpr, vb_tpr, thresh = roc_curve(Y, pvb)
    ax.plot(mc_fpr, mc_tpr, label="MCMC", linewidth=3)
    ax.plot(vb_fpr, vb_tpr, label="VI", linewidth=3)
    if compare_to_photo:
        photo_fpr = np.sum((photodf.is_star) & (~truedf.is_star)) / np.sum(~truedf.is_star)
        photo_tpr = np.sum((photodf.is_star) & (truedf.is_star)) / np.sum(truedf.is_star)
        ax.scatter(photo_fpr, photo_tpr, marker="x", color='red', linewidth=2, s=35, label="Photo")

    ax.legend(fontsize=fontsize)
    ax.set_xlim(0, .4)
    ax.set_ylim(.6, 1.)
    ax.set_xlabel("False Positive Rate", fontsize=fontsize)
    ax.set_ylabel("True Positive Rate")
    fig.savefig(os.path.join(results_dir, "pstar_roc_comparison.png"), bbox_inches='tight', dpi=200)

    # compute AUC for MC and VB
    def make_mc_vb_auc_df():
        mc_aucs, vb_aucs = [], []
        for i in range(5000):
            idx = np.random.choice(len(Y), size=len(Y))
            mc_aucs.append(roc_auc_score(Y[idx], pmc[idx]))
            vb_aucs.append(roc_auc_score(Y[idx], pvb[idx]))
        return pd.DataFrame({'inference method': ['MCMC']*len(mc_aucs) + ['VI']*len(vb_aucs),
                             'AUC'   : np.concatenate([mc_aucs, vb_aucs])})

    aucdf = make_mc_vb_auc_df()
    mc_auc = roc_auc_score(Y, pmc)
    vb_auc = roc_auc_score(Y, pvb)
    mc_aucs = aucdf['AUC'][aucdf['inference method'] == 'MCMC']
    vb_aucs = aucdf['AUC'][aucdf['inference method'] == 'VI']
    print("======= Star vs. Galaxy AUC Scores ==========")
    print("   MC : %2.4f  [%2.4f,  %2.4f] "%(mc_auc, np.percentile(mc_aucs, 2.5), np.percentile(mc_aucs, 97.5)))
    print("   VB : %2.4f  [%2.4f,  %2.4f] "%(vb_auc, np.percentile(vb_aucs, 2.5), np.percentile(vb_aucs, 97.5)))

    fig, ax = plt.figure(figsize=(6,4)), plt.gca()
    sns.violinplot(x='inference method', y='AUC', data=aucdf)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "pstar_auc_comparison.png"), bbox_inches='tight', dpi=200)


def make_mcmc_vb_uncertainty_comparison_plots(results_dir, source_type="star", param_name="log_flux_r"):
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
        fig, ax = plt.figure(figsize=(6,3)), plt.gca()
        lo, hi = min(mcmu - 2.75*mcscale, vbmu-2.75*vbscale, true_val), \
                 max(mcmu + 2.75*mcscale, vbmu+2.75*vbscale, true_val)
        xgrid = np.linspace(lo, hi, 1000)
        ax.plot(xgrid, norm.pdf(xgrid, vbmu, vbscale), label="VI", linewidth=2, c=cvb)
        ax.fill_between(xgrid, norm.pdf(xgrid, vbmu, vbscale), alpha=.5, color=cvb)
        ax.plot(xgrid, norm.pdf(xgrid, mcmu, mcscale), "--", label="MCMC", linewidth=2, c=cmc)
        ax.fill_between(xgrid, norm.pdf(xgrid, mcmu, mcscale), alpha=.5, color=cmc)
        ax.scatter(true_val, 0., s=200, linewidth=3, marker='x', label="true", c=ctrue)
        ax.set_xlabel("log brightness")
        ax.legend(fontsize=fontsize)
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


##################
# helper error   #
##################
def angular_separation(lam1, phi1, lam2, phi2):
    dlam = lam2 - lam1
    sin_dlam = np.sin(np.deg2rad(dlam))
    cos_dlam = np.cos(np.deg2rad(dlam))
    sin_phi1 = np.sin(np.deg2rad(phi1))
    sin_phi2 = np.sin(np.deg2rad(phi2))
    cos_phi1 = np.cos(np.deg2rad(phi1))
    cos_phi2 = np.cos(np.deg2rad(phi2))
    hp = np.hypot(cos_phi2 * sin_dlam,
                  cos_phi1 * sin_phi2 - sin_phi1 *cos_phi2 * cos_dlam)
    at = np.arctan2(hp, sin_phi1*sin_phi2 + cos_phi1*cos_phi2*cos_dlam)
    return np.rad2deg(at)

def position_error(true_ra, true_dec, pred_ra, pred_dec):
    ARCSEC_PER_DEGREE = 3600
    SDSS_ARCSEC_PER_PIXEL = 0.396
    return (ARCSEC_PER_DEGREE / SDSS_ARCSEC_PER_PIXEL) * \
            angular_separation(true_ra, true_dec, pred_ra, pred_dec)


##############
# run script #
##############
if __name__=="__main__":
    main()

