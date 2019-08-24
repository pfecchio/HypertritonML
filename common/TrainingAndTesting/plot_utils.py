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
from sklearn.metrics import auc, roc_curve


# plot the BDT score distribution in the train and in the test set for both signal and background
def plot_output_train_test(
        clf, x_train, y_train, x_test, y_test, ct_range=[0, 100],
        pt_range=[2, 3],
        cent_range=[0, 10],
        draw=True, model='xgb', features=None, raw=True, bins=80, figsize=(7.5, 5),
        path='', location='best', **kwds):
    """
    model could be 'xgb' or 'sklearn'
    """

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
    plt.xlabel("BDT output", fontsize=13, ha='right', position=(1, 20))
    plt.ylabel(r'                                Counts (arb. units)', fontsize=13)
    plt.legend(loc=location, frameon=False, fontsize=12)

    # plt.xlim(-2, 2)
    # plt.ylim(10 ** -5, 10 ** 2)

    # aliceleg = r'$\mathrm{\ \ \ \mathbf{ALICE \ Simulation}}$' + '\n' + r'Pb-Pb $\sqrt{s_{\mathrm{NN}}}$ = 5.02 TeV      '
    # plt.text(-14, 5, aliceleg, size='x-large')

    fig_name = 'Plot_ct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_cen_{:.2f}_{:.2f}'.format(
        ct_range[0], ct_range[1], pt_range[0], pt_range[1], cent_range[0], cent_range[1])

    plt.savefig('{}/{}.pdf'.format(path, fig_name), dpi=500, transparent=True)
    plt.close()
