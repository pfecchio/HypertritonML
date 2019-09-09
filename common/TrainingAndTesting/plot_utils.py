import io
import os
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

import pandas as pd
import xgboost as xgb
from pandas.core.index import Index
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import auc, roc_curve

from ROOT import TH1D,TCanvas,TFile,gStyle
from array import array
import math
import numpy as np


# plot the BDT score distribution in the train and in the test set for both signal and background
def plot_output_train_test(
        clf, x_train, y_train, x_test, y_test, ct_range=[0, 100],
        pt_range=[0, 100],
        cent_class=[0, 100],
        model='xgb', features=None, raw=True, bins=80, figsize=(7.5, 5),
        location='best', mode=3, **kwds):
    '''
    model could be 'xgb' or 'sklearn'
    '''

    prediction = []
    for x, y in ((x_train, y_train), (x_test, y_test)):
        if model == 'xgb':
            d1 = clf.predict(xgb.DMatrix(x[y > 0.5], feature_names=features), output_margin=raw)
            d2 = clf.predict(xgb.DMatrix(x[y < 0.5], feature_names=features), output_margin=raw)
        elif model == 'sklearn':
            d1 = clf.decision_function(x[y > 0.5]).ravel()
            d2 = clf.decision_function(x[y < 0.5]).ravel()
        else:
            print('Error: wrong model type used')
            return
        prediction += [d1, d2]

    print(stats.ks_2samp(d1, d2))

    low = min(np.min(d) for d in prediction)
    high = max(np.max(d) for d in prediction)
    low_high = (low, high)

    plt.figure(figsize=figsize)
    plt.hist(prediction[1], color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True, log=True, label='Background pdf Training Set')
    plt.hist(prediction[0], color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True, log=True, label='Signal pdf Training Set')

    hist, bins = np.histogram(prediction[2], bins=bins, range=low_high, density=True)
    scale = len(prediction[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='Background pdf Test Set')

    hist, bins = np.histogram(prediction[3], bins=bins, range=low_high, density=True)
    scale = len(prediction[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='Signal pdf Test Set')

    # plt.gcf().subplots_adjust(left=0.14)
    plt.xlabel('BDT output', fontsize=13, ha='right', position=(1, 20))
    plt.ylabel(r'                                Counts (arb. units)', fontsize=13)
    plt.legend(loc=location, frameon=False, fontsize=12)

    # plt.xlim(-2, 2)
    # plt.ylim(10 ** -5, 10 ** 2)

    # aliceleg = r'$\mathrm{\ \ \ \mathbf{ALICE \ Simulation}}$' + '\n' + r'Pb-Pb $\sqrt{s_{\mathrm{NN}}}$ = 5.02 TeV      '
    # plt.text(-14, 5, aliceleg, size='x-large')

    fig_score_path = os.environ['HYPERML_FIGURES_{}'.format(mode)]+'/TrainTest'
    if not os.path.exists(fig_score_path):
        os.makedirs(fig_score_path)

    fig_name = 'BDTscorePDF_ct{}{}_pT{}{}_cen{}{}'.format(
        ct_range[0], ct_range[1], pt_range[0], pt_range[1], cent_class[0], cent_class[1])

    plt.savefig('{}/{}.pdf'.format(fig_score_path, fig_name), dpi=500, transparent=True)
    plt.close()


def plot_distr(df, column=None, figsize=None, bins=50, fig_name='features.pdf', mode=3, **kwds):
    """Build a DataFrame and create two dataset for signal and bkg

    Draw histogram of the DataFrame's series comparing the distribution
    in `data1` to `data2`.

    X: data vector
    y: class vector
    column: string or sequence
        If passed, will be used to limit data to a subset of columns
    figsize : tuple
        The size of the figure to create in inches by default
    bins: integer, default 10
        Number of histogram bins to be used
    kwds : other plotting keyword arguments
        To be passed to hist function
    """

    data1 = df[df.y < 0.5]
    data2 = df[df.y > 0.5]

    if column is not None:
        if not isinstance(column, (list, np.ndarray, Index)):
            column = [column]
        data1 = data1[column]
        data2 = data2[column]

    if figsize is None:
        figsize = [15, 10]

    axes = data1.hist(column=column, color='blue', alpha=0.5, bins=bins, figsize=figsize,
                      label='Background', density=True, grid=False, log=True,  **kwds)
    axes = axes.flatten()
    axes = axes[:len(column)]
    data2.hist(ax=axes, column=column, color='red', alpha=0.5, bins=bins, label='Signal',
               density=True, grid=False, log=True, **kwds)[0].legend()
    for a in axes:
        a.set_ylabel('Counts (arb. units)')

    fig_features_path = os.environ['HYPERML_FIGURES_{}'.format(mode)]+'/TrainTest'
    if not os.path.exists(fig_features_path):
        os.makedirs(fig_features_path)

    plt.savefig('{}/{}.pdf'.format(fig_features_path, fig_name), dpi=500, transparent=True)
    plt.close()


def plot_corr(df, columns, mode=3, **kwds):
    """Calculate pairwise correlation between features.
    Extra arguments are passed on to DataFrame.corr()
    """
    col = columns+['y']
    df = df[col]

    data_sig = df[df.y > 0.5].drop('y', 1)
    data_bkg = df[df.y < 0.5].drop('y', 1)

    corrmat_sig = data_sig.corr(**kwds)
    corrmat_bkg = data_bkg.corr(**kwds)

    t = r'$\mathrm{\ \ \ ALICE \ Simulation}$ Pb-Pb $\sqrt{s_{\mathrm{NN}}}$ = 5.02 TeV'
    fig = plt.figure(figsize=(20, 10))
    # plt.title(t,y=1.08,fontsize=16)
    plt.suptitle(t, fontsize=18, ha='center')
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.15, share_all=True,
                     cbar_location='right', cbar_mode='single', cbar_size='7%', cbar_pad=0.15)

    opts = {'cmap': plt.get_cmap('coolwarm'), 'vmin': -1, 'vmax': +1, 'snap': True}

    ax1 = grid[0]
    ax2 = grid[1]
    heatmap1 = ax1.pcolor(corrmat_sig, **opts)
    heatmap2 = ax2.pcolor(corrmat_bkg, **opts)
    ax1.set_title('Signal', fontsize=14, fontweight='bold')
    ax2.set_title('Background', fontsize=14, fontweight='bold')

    lab = corrmat_sig.columns.values
    # lab = [r'$\it{M}_{\mathrm{He}^{3}\pi^{-}}$', r'n$\sigma_{\mathrm{TPC}}\ \mathrm{He}^{3}$',
    #        r'$\mathrm{V}_{0} \ p_{\mathrm{T}}\ (\mathrm{GeV}/c)$', r'n$_{cluster\ \mathrm{TPC}}\ \mathrm{He}^{3}$',
    #        r'$\alpha$-armenteros', r'L/$p$ ($\frac{cm}{\mathrm{Gev}/c}$)',
    #        r'$\mathrm{DCA}_{\mathrm{V_{0}\ tracks}} ($cm$)$', r'$\cos{(\theta_{pointing})}$']
    for ax in (ax1,):
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(lab)), minor=False)
        ax.set_yticks(np.arange(len(lab)), minor=False)
        ax.set_xticklabels(lab, minor=False, ha='left', rotation=90, fontsize=10)
        ax.set_yticklabels(lab, minor=False, va='bottom', fontsize=10)
        ax.tick_params(axis='both', which='both', direction="in")

        for tick in ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('center')

    for ax in (ax2,):
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(lab)), minor=False)
        ax.set_yticks(np.arange(len(lab)), minor=False)
        ax.set_xticklabels(lab, minor=False, ha='left', rotation=90, fontsize=10)
        ax.tick_params(axis='both', which='both', direction="in")
        for tick in ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('center')

    ax1.cax.colorbar(heatmap1)
    ax1.cax.toggle_label(True)

    fig_corr_path = os.environ['HYPERML_FIGURES_{}'.format(mode)]+'/TrainTest'
    if not os.path.exists(fig_corr_path):
        os.makedirs(fig_corr_path)

    fig_name = 'correlations.pdf'

    plt.savefig('{}/{}.pdf'.format(fig_corr_path, fig_name), dpi=500, transparent=True)
    plt.close()


def plot_bdt_eff(threshold, eff_sig, mode, ct_range=[0, 100], pt_range=[0, 100], cent_class=[0, 100]):
    plt.plot(threshold, eff_sig, 'r.', label='Signal efficiency')
    plt.legend()
    plt.xlabel('BDT Score')
    plt.ylabel('Efficiency')
    plt.title('Efficiency vs Score')
    plt.grid()

    fig_eff_path = os.environ['HYPERML_FIGURES_{}'.format(mode)]+'/Efficiency'
    if not os.path.exists(fig_eff_path):
        os.makedirs(fig_eff_path)

    fig_name = '/BDTeffct{}{}_pT{}{}_cen{}{}.pdf'.format(
        ct_range[0], ct_range[1], pt_range[0], pt_range[1], cent_class[0], cent_class[1])
    plt.savefig(fig_eff_path + fig_name)
    plt.close()


def plot_efficiency_significance(mode, tsd, significance, efficiency, data_range_array):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'

    ax1.set_xlabel('BDT Score')
    ax1.set_ylabel('Significance', color=color)
    ax1.plot(tsd, significance, color=color)
    ax1.tick_params(axis='y', labelcolor=color, direction='in')

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('BDT efficiency', color=color)  # we already handled the x-label with ax1
    ax2.plot(tsd, efficiency, color=color)
    ax2.tick_params(axis='y', labelcolor=color, direction='in')

    fig.tight_layout()

    fig_eff_path = os.environ['HYPERML_FIGURES_{}'.format(mode)]+'/Significance'
    if not os.path.exists(fig_eff_path):
        os.makedirs(fig_eff_path)

    fig_name = '/sign_eff_ct{}{}_pT{}{}_cen{}{}.pdf'.format(
        data_range_array[0], data_range_array[1], data_range_array[2], data_range_array[3], data_range_array[4], data_range_array[5])
    plt.savefig(fig_eff_path + fig_name)
    plt.close()

def plot_eff_ct(analysis, file_name, pt_range = [2,10], cent_class = [0,90], ct_bins = [[0,2],[2,4],[4,6],[6,8],[8,10],[10,14],[14,18],[18,23],[23,28]]):
  
  Analysis = GeneralizedAnalysis(2,os.environ['HYPERML_TABLES_2']+'/SignalTable.root',os.environ['HYPERML_TABLES_2']+'/DataTable.root','2<=HypCandPt<=10','(InvMass<2.98 or InvMass>3.005) and HypCandPt<=10',cent_class=[[0,90]])
 
  results = TFile(os.environ['HYPERML_DATA_2']+"/"+file_name,"update")
  results.cd()
  ct_binning = [ct_bins[0][0]]
  for ct in ct_bins:
    ct_binning.append(ct[1])
  ct_binning = array("d",ct_binning)
  cv = TCanvas("presel_efficiency")
  histo_eff = TH1D("histo_eff_ct",";ct (cm);preselection efficiency",len(ct_binning)-1,ct_binning)
  gStyle.SetOptStat(0)
  gStyle.SetOptFit(0)
  for ct in Ct_bins:
    Effp = (Analysis.preselection_efficiency(ct_range=ct,pt_range=pt_range,cent_class=cent_class))

    selection = '{}<ct<{} and {}<centrality<{} and {}<pT<{}'.format(ct[0],ct[1],pt_range[0],pt_range[1],cent_class[0],cent_class[1])
    n_gen = sum(Analysis.df_generated.query(selection)['y'])

    errEff = math.sqrt((1-Effp)*Effp/n_gen)

    ct_index=ct_bins.index(ct)
    histo_eff.SetBinContent(ct_index+1,Effp)
    histo_eff.SetBinError(ct_index+1,errEff)

  histo_eff.Draw()
  cv.Write()
  cv.SaveAs(os.environ['HYPERML_DATA_2']+'/presel_efficiency.pdf')

def plot_significance_scan(
        max_index, significance, significance_error, expected_signal, bkg_df, score_list, data_range_array, bin_cent,
        n_ev, mode, custom=True):
    label = 'Significance'
    if custom:
        label = label + ' x Efficiency'

    raw_yield = expected_signal[max_index]
    max_score = score_list[max_index]

    selected_bkg = bkg_df.query('Score>@max_score')

    signal_counts_norm = norm.pdf(bin_cent, loc=2.992, scale=0.0025)
    signal_counts = raw_yield * signal_counts_norm / sum(signal_counts_norm)

    bkg_counts, bins = np.histogram(selected_bkg['InvMass'], bins=25, range=[2.96, 3.05])

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    side_map = (bin_centers < 2.98) + (bin_centers > 3.005 )
    bins_side = bin_centers[side_map]
    mass_map = np.logical_not(side_map)

    bkg_side_counts = bkg_counts[side_map]

    bkg_roi_shape = np.polyfit(bins_side, bkg_side_counts, 2)
    bkg_roi_counts = np.polyval(bkg_roi_shape, bin_centers)

    tot_counts = (bkg_roi_counts + signal_counts)[mass_map]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].set_xlabel('Score')
    axs[0].set_ylabel(label)
    axs[0].tick_params(axis='x', direction='in')
    axs[0].tick_params(axis='y', direction='in')
    axs[0].plot(score_list, significance, 'b', label='Expected {}'.format(label))

    significance = np.asarray(significance)
    significance_error = np.asarray(significance_error)

    low_limit = significance - significance_error
    up_limit = significance + significance_error

    axs[0].fill_between(score_list, low_limit, up_limit, facecolor='deepskyblue', label=r'$ \pm 1\sigma$')
    axs[0].grid()
    axs[0].legend(loc='upper left')

    bkg_side_error = np.sqrt(bkg_side_counts)
    tot_counts_error = np.sqrt(tot_counts)

    bins_mass = bin_centers[mass_map]

    axs[1].errorbar(bins_side, bkg_side_counts, yerr=5,
                    fmt='.', ecolor='k', color='b', elinewidth=1., label='Data')
    axs[1].errorbar(bins_mass, tot_counts, yerr=tot_counts_error,
                    fmt='.', ecolor='k', color='r', elinewidth=1., label='Pseudodata')
    axs[1].plot(bin_centers, bkg_roi_counts, 'g-', label='Background fit')

    x = np.linspace(2.9923 - 3 * 0.0025, 2.9923 + 3 * 0.0025, 1000)
    gauss_signal_counts = norm.pdf(x, loc=2.992, scale=0.0025)
    gauss_signal_counts = (raw_yield / sum(signal_counts_norm)) * gauss_signal_counts + np.polyval(bkg_roi_shape, x)

    axs[1].plot(x, gauss_signal_counts, 'y', color='orange', label='Signal model (Gauss)')
    axs[1].set_xlabel(r'$m_{\ ^{3}He+\pi^{-}}$')
    axs[1].set_ylabel(r'Events /  $3.6\ \rm{MeV}/c^{2}$')
    axs[1].tick_params(axis='x', direction='in')
    axs[1].tick_params(axis='y', direction='in')
    axs[1].legend(loc='best', frameon=False)
    plt.ylim(bottom=0)

    s = sum(tot_counts) - sum(bkg_roi_counts[mass_map])
    b = sum(bkg_roi_counts[mass_map])

    sign_score = s / np.sqrt(s + b)

    plt.suptitle(r'%1.f$\leq ct \leq$%1.f %1.f$\leq \rm{p}_{T} \leq$%1.f  Cut Score=%0.2f  Significance=%0.2f  Raw yield=%0.2f' % (
        data_range_array[0], data_range_array[1], data_range_array[2], data_range_array[3], max_score,  sign_score, raw_yield))

    # text = '\n'.join(
    #     r'%1.f GeV/c $ \leq \rm{p}_{T} < $ %1.f GeV/c ' % (data_range_array[0], data_range_array[1]),
    #     r' Significance/Sqrt(Events) = %0.4f$x10^{-4}$' % (max_significance / np.sqrt(n_ev) * 1e4))

    # props = dict(boxstyle='round', facecolor='white', alpha=0)

    # axs[1].text(0.37, 0.95, text, transform=axs[1].transAxes, verticalalignment='top', bbox=props)

    fig_name = 'Significance_ct{}{}_pT{}{}_cen{}{}.pdf'.format(
        data_range_array[0],
        data_range_array[1],
        data_range_array[2],
        data_range_array[3],
        data_range_array[4],
        data_range_array[5])

    fig_sig_path = os.environ['HYPERML_FIGURES_{}'.format(mode)]+'/Significance'
    if not os.path.exists(fig_sig_path):
        os.makedirs(fig_sig_path)

    plt.savefig(fig_sig_path + '/' + fig_name)
    plt.close()
