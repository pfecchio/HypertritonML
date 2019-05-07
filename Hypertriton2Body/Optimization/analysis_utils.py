import io
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
from pandas.core.index import Index
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_curve, auc

# Data Visualization Functions
###########################################################################################
def plot_distr(df, column=None, figsize=None, bins=25, **kwds):
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
                      label="Background", density=True, grid=False, **kwds)
    axes = axes.flatten()
    axes = axes[:len(column)]
    data2.hist(ax=axes, column=column, color='red', alpha=0.5, bins=bins, label="Signal",
               density=True, grid=False, **kwds)[0].legend()
    for a in axes:
        a.set_ylabel("Arbitrary Units")


def plot_corr(df, columns, title, figsize=(6, 6), **kwds):
    """Calculate pairwise correlation between features.

    Extra arguments are passed on to DataFrame.corr()
    """
    col=columns+['y']
    df=df[col]
    if title == "Signal":
        data = df[df.y > 0.5].drop('y', 1)
    elif title == "Background":
        data = df[df.y < 0.5].drop('y', 1)

    corrmat = data.corr(**kwds)
    _, ax1 = plt.subplots(ncols=1, figsize=figsize)
    opts = {'cmap': plt.get_cmap("coolwarm"), 'vmin': -1, 'vmax': +1, 'snap': True}
    heatmap1 = ax1.pcolor(corrmat, **opts)
    plt.colorbar(heatmap1, ax=ax1)
    ax1.set_title(title, fontsize=22)

    labels = corrmat.columns.values
    for ax in (ax1,):
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels)), minor=False)
        ax.set_yticks(np.arange(len(labels)), minor=False)
        ax.set_xticklabels(labels, minor=False, ha='left', rotation=90, fontsize=15)
        ax.set_yticklabels(labels, minor=False, va='bottom', fontsize=15)

        for tick in ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('center')

    plt.tight_layout()


def plot_roc(y_truth, model_decision):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_truth, model_decision)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.4f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right")
    plt.grid()


def plot_output_train_test(clf, x_train, y_train, x_test, y_test, model='xgb', branch_names=None,
                           raw=True, bins=50, figsize=(7, 5), location='best', **kwds):
    """
    model could be 'xgb' or 'sklearn'
    """
    decisions = []
    for x, y in ((x_train, y_train), (x_test, y_test)):
        if model == 'xgb':
            d1 = clf.predict(xgb.DMatrix(x[y > 0.5], feature_names=branch_names), output_margin=raw)
            d2 = clf.predict(xgb.DMatrix(x[y < 0.5], feature_names=branch_names), output_margin=raw)
        elif model == 'sklearn':
            d1 = clf.decision_function(x[y > 0.5]).ravel()
            d2 = clf.decision_function(x[y < 0.5]).ravel()
        else:
            print('Error: wrong model typr used')
            return
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low, high)

    plt.figure(figsize=figsize)
    plt.hist(decisions[0], color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True, label='S, train', **kwds)
    plt.hist(decisions[1], color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True, label='B, train', **kwds)

    hist, bins = np.histogram(decisions[2], bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S, test')

    hist, bins = np.histogram(decisions[3], bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B, test')

    plt.gcf().subplots_adjust(left=0.14)
    plt.xlabel("BDT output", fontsize=15)
    plt.ylabel("Arbitrary units", fontsize=15)
    plt.legend(loc=location, frameon=False, fontsize=15)


def plot_feature_imp(model, imp_list=None, line_pos=None):

    n_plots = len(imp_list)
    _, ax1 = plt.subplots(ncols=n_plots, figsize=(20, 10), squeeze=False)
    ax1 = ax1[0]

    for imp_type, axc in zip(imp_list, ax1):
        feat_imp = pd.Series(model.get_score(importance_type=imp_type))
        feat_imp = feat_imp * 1. / feat_imp.sum()
        feat_imp.plot(ax=axc, kind='bar', fontsize=25)
        axc.set_ylabel(f'Relative {imp_type}', fontsize=35)
        axc.set_xticklabels(axc.get_xticklabels())

        if line_pos is not None:
            axc.axhline(y=line_pos, color='r', linestyle='-', linewidth=6)

    plt.tight_layout(w_pad=10)


#Training Functions
###########################################################################################
def gs_1par(gs_dict, par_dict, train_data, num_rounds, seed, folds, metrics, n_early_stop):

    fp_dict = gs_dict['first_par']
    gs_params = fp_dict['par_values']

    max_auc = 0.
    max_std = 0.
    best_params = None
    for val in gs_params:
        # Update our parameters
        par_dict[fp_dict['name']] = val

        # Run CV
        trap = io.StringIO()
        with redirect_stdout(trap):
            cv_results = xgb.cv(par_dict, train_data, num_boost_round=num_rounds, seed=seed,
                                folds=folds, metrics=metrics, early_stopping_rounds=n_early_stop)

        # Update best AUC
        mean_auc = cv_results['test-auc-mean'].max()
        boost_rounds = cv_results['test-auc-mean'].idxmax()
        mean_std = cv_results['test-auc-std'][boost_rounds]
        if mean_auc > max_auc:
            max_auc = mean_auc
            max_std = mean_std
            best_params = (val, boost_rounds)
    return (best_params)


def gs_2par(gs_dict, par_dict, train_data, num_rounds, seed, folds, metrics, n_early_stop):

    fp_dict = gs_dict['first_par']
    sp_dict = gs_dict['second_par']
    gs_params = [(first_val, second_val)
                 for first_val in fp_dict['par_values']
                 for second_val in sp_dict['par_values']
                ]

    max_auc = 0.
    max_std = 0.
    best_params = None
    for first_val, second_val in gs_params:
        # Update our parameters
        par_dict[fp_dict['name']] = first_val
        par_dict[sp_dict['name']] = second_val

        # Run CV
        trap = io.StringIO()
        with redirect_stdout(trap):
            cv_results = xgb.cv(par_dict, train_data, num_boost_round=num_rounds, seed=seed,
                                folds=folds, metrics=metrics, early_stopping_rounds=n_early_stop)

        # Update best AUC
        mean_auc = cv_results['test-auc-mean'].max()
        boost_rounds = cv_results['test-auc-mean'].idxmax()
        mean_std = cv_results['test-auc-std'][boost_rounds]
        if mean_auc > max_auc:
            max_auc = mean_auc
            max_std = mean_std
            best_params = (first_val, second_val,boost_rounds)
    return (best_params)

def EfficiencyVsCuts(df,plot_ext=False):
    if plot_ext==True:
        cuts=np.linspace(-15,15,200)
    else:    
        cuts=np.linspace(-3,12.5,100)
    eff_s=[]
    eff_b=[]
    den_s=sum(df['y'])
    den_b=len(df)-den_s
    for i in cuts:
        cut_df=df.query('Score>@i')['y']
        num_s=np.sum(cut_df)
        num_b=len(cut_df)-num_s
        eff_s.append(num_s/den_s)
        eff_b.append(num_b/den_b)
    h=np.polyfit(cuts,eff_s,7)
    y=np.polyval(h,cuts)
    plt.plot(cuts,y,'b-',label='Fit efficiency')
    plt.plot(cuts,eff_s,'r.',label='Signal efficiency')
    #plt.plot(cuts,eff_b,'b.-',label='bkg_dist')
    plt.legend()
    plt.xlabel('Score')
    plt.ylabel('Efficiency')
    plt.title('Efficiency vs Score')
    plt.grid()
    if plot_ext==False:
        return eff_s
