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


# plot the BDT score distribution in the train and in the test set for both signal and background
def plot_output_train_test(
        clf, x_train, y_train, x_test, y_test, ct_range=[0, 100],
        pt_range=[0, 100],
        cent_range=[0, 100], model='xgb', features=None, raw=True, bins=80, figsize=(7.5, 5),
        path='', location='best', **kwds):
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
             histtype='stepfilled', density=True, label='Signal pdf Training Set', **kwds)
    plt.hist(prediction[0], color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True, label='Background pdf Training Set', **kwds)

    hist, bins = np.histogram(prediction[2], bins=bins, range=low_high, density=True)
    scale = len(prediction[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='Signal pdf Test Set')

    hist, bins = np.histogram(prediction[3], bins=bins, range=low_high, density=True)
    scale = len(prediction[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='Background pdf Test Set')

    # plt.gcf().subplots_adjust(left=0.14)
    plt.xlabel('BDT output', fontsize=13, ha='right', position=(1, 20))
    plt.ylabel(r'                                Counts (arb. units)', fontsize=13)
    plt.legend(loc=location, frameon=False, fontsize=12)

    # plt.xlim(-2, 2)
    # plt.ylim(10 ** -5, 10 ** 2)

    # aliceleg = r'$\mathrm{\ \ \ \mathbf{ALICE \ Simulation}}$' + '\n' + r'Pb-Pb $\sqrt{s_{\mathrm{NN}}}$ = 5.02 TeV      '
    # plt.text(-14, 5, aliceleg, size='x-large')

    fig_name = 'BDTscorePDF_ct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_cen_{:.2f}_{:.2f}'.format(
        ct_range[0], ct_range[1], pt_range[0], pt_range[1], cent_range[0], cent_range[1])

    plt.savefig('{}/{}.pdf'.format(path, fig_name), dpi=500, transparent=True)
    plt.close()


def plot_bdt_eff(threshold, eff_sig, mode, ct_range=[0, 100], pt_range=[0, 100], cent_range=[0, 100]):
    plt.plot(threshold, eff_sig, 'r.', label='Signal efficiency')
    plt.legend()
    plt.xlabel('BDT Score')
    plt.ylabel('Efficiency')
    plt.title('Efficiency vs Score')
    plt.grid()

    fig_eff_path = os.environ['HYPERML_FIGURES_{}'.format(mode)]+'/Efficiency'
    if not os.path.exists(fig_eff_path):
        os.makedirs(fig_eff_path)

    fig_name = '/BDTeffct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_cen_{:.2f}_{:.2f}.pdf'.format(
        ct_range[0], ct_range[1], pt_range[0], pt_range[1], cent_range[0], cent_range[1])
    plt.savefig(fig_eff_path + fig_name)
    plt.close()


def plot_significance_scan(
        max_index, significance, significance_error, expected_signal, bkg_df, score_list, data_range_array, bin_cent,
        n_ev, mode, custom=True):
    label = 'Significance'
    if custom:
        label = label + ' x Efficiency'

    max_significance = significance[max_index]
    raw_yield = expected_signal[max_index]
    max_score = score_list[max_index]

    selected_bkg = bkg_df.query('Score>@max_score')

    signal_counts_norm = norm.pdf(bin_cent, loc=2.992, scale=0.0025)
    signal_counts = raw_yield * signal_counts_norm / sum(signal_counts_norm)

    bkg_side_counts, bins = np.histogram(selected_bkg['InvMass'], bins=30, range=[2.96, 3.05])

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    side_map = (bin_centers < 2.9923 - 3 * 0.0025) + (bin_centers > 2.9923 + 3 * 0.0025)
    bins_side = bin_centers[side_map]
    mass_map = np.logical_not(side_map)

    bkg_roi_shape = np.polyfit(bins_side, bkg_side_counts[side_map], 2)
    bkg_roi_counts = np.polyval(bkg_roi_shape, bin_centers)
    tot_counts = bkg_roi_counts + signal_counts

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].set_xlabel('Score')
    axs[0].set_ylabel(label)
    axs[0].tick_params(axis='x', direction='in')
    axs[0].tick_params(axis='y', direction='in')
    axs[0].plot(score_list, significance, 'b', label='Expected significance')

    significance = np.asarray(significance)
    significance_error = np.asarray(significance_error)

    low_limit = significance - significance_error
    up_limit = significance + significance_error

    axs[0].fill_between(score_list, low_limit, up_limit, facecolor='deepskyblue', label=r'$ \pm 1\sigma$')
    axs[0].grid()
    axs[0].legend(loc='upper left')
    plt.suptitle(r'%1.f $ \leq \rm{p}_{T} \leq $ %1.f, Cut Score = %0.2f, Significance/Events = %0.4f$x10^{-4}$, %s = %0.2f , Raw yield = %0.2f' % (
        data_range_array[2], data_range_array[3], max_score, (max_significance / np.sqrt(n_ev) * 1e4), label, max_significance, raw_yield))

    bkg_side_error = np.sqrt(bkg_side_counts[side_map])
    tot_counts_error = np.sqrt(tot_counts[mass_map])
    mass_map = bin_centers[mass_map]
    bin_centers_map = bin_centers[side_map]
    bkg_roi_counts_map = bkg_roi_counts[side_map]

    axs[1].errorbar(bin_centers_map, bkg_side_error, yerr=bkg_side_error,
                    fmt='.', ecolor='k', color='b', elinewidth=1., label='Data')
    axs[1].errorbar(mass_map, tot_counts_error, yerr=tot_counts_error,
                    fmt='.', ecolor='k', color='r', elinewidth=1., label='Pseudodata')
    axs[1].plot(bin_centers_map, bkg_roi_counts_map, 'g-', label='Background fit')

    x = np.linspace(2.9923 - 3 * 0.0025, 2.9923 + 3 * 0.0025, 1000)
    gauss_signal_counts = norm.pdf(x, loc=2.992, scale=0.0025)
    gauss_signal_counts = (raw_yield / sum(signal_counts_norm)) * gauss_signal_counts + np.polyval(bkg_roi_shape, x)

    axs[1].plot(x, gauss_signal_counts, 'y', color='orange', label='Signal model (Gauss)')
    axs[1].set_xlabel(r'$m_{\ ^{3}He+\pi^{-}}$')
    axs[1].set_ylabel(r'Events /  $3.6\ \rm{MeV}/c^{2}$')
    axs[1].tick_params(axis='x', direction='in')
    axs[1].tick_params(axis='y', direction='in')
    axs[1].legend(loc=(0.37, 0.47))
    plt.ylim(bottom=0)

    # text = '\n'.join(
    #     r'%1.f GeV/c $ \leq \rm{p}_{T} < $ %1.f GeV/c ' % (data_range_array[0], data_range_array[1]),
    #     r' Significance/Sqrt(Events) = %0.4f$x10^{-4}$' % (max_significance / np.sqrt(n_ev) * 1e4))

    # props = dict(boxstyle='round', facecolor='white', alpha=0)

    # axs[1].text(0.37, 0.95, text, transform=axs[1].transAxes, verticalalignment='top', bbox=props)

    fig_name = 'Significance_ct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_Cen_{:.2f}_{:.2f}.pdf'.format(
        data_range_array[0],
        data_range_array[1],
        data_range_array[2],
        data_range_array[3],
        data_range_array[4],
        data_range_array[5])

    fig_sig_path = os.environ['HYPERML_FIGURES_{}'.format(mode)]+'/Significance'
    if not os.path.exists(fig_sig_path):
        os.makedirs(fig_sig_path)

    fig_name = '/BDTeffct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_cen_{:.2f}_{:.2f}.pdf'.format(
        data_range_array[0],
        data_range_array[1],
        data_range_array[2],
        data_range_array[3],
        data_range_array[4],
        data_range_array[5])
    plt.savefig(fig_sig_path + fig_name)
    plt.close()
