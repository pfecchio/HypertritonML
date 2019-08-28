import io
import os
from contextlib import redirect_stdout

import numpy as np

import xgboost as xgb
from ROOT import TF1, TFile, gDirectory


# target function for the bayesian hyperparameter optimization
def evaluate_hyperparams(
        dtrain, reg_params, eta, min_child_weight, max_depth, gamma, subsample, colsample_bytree,
        scale_pos_weight, num_rounds=100, es_rounds=2, nfold=3, round_score_list=[]):
    params = {'eval_metric': 'auc',
              'eta': eta,
              'min_child_weight': int(min_child_weight),
              'max_depth': int(max_depth),
              'gamma': gamma,
              'subsample': subsample,
              'colsample_bytree': colsample_bytree,
              'scale_pos_weight': scale_pos_weight}
    params = {**reg_params, **params}

    # Use around 1000 boosting rounds in the full model
    cv_result = xgb.cv(params, dtrain, num_boost_round=num_rounds, early_stopping_rounds=es_rounds, nfold=nfold)

    best_boost_rounds = cv_result['test-auc-mean'].idxmax()
    best_score = 1000 * (cv_result['test-auc-mean'].max() - 0.999)

    round_score_list.append(tuple([best_score, best_boost_rounds]))

    return best_score


def gs_1par(gs_dict, par_dict, train_data, num_rounds, seed, folds, metrics, n_early_stop):

    fp_dict = gs_dict['first_par']
    gs_params = fp_dict['par_values']

    max_auc = 0.
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
            best_params = (first_val, second_val, boost_rounds)
    return (best_params)


def expected_signal_raw(pt_range, cent_class):
    bw_ = os.environ['HYPERML_UTILS']
    bw_file = TFile(os.environ['HYPERML_UTILS'] + '/BlastWaveFits.root')

    scale_factor = [3.37e-5, 1.28e-5, 0.77e-5, 0.183e-5]

    if cent_class == 2:
        bw1 = bw_file.Get("BlastWave/BlastWave1")
        bw2 = bw_file.Get("BlastWave/BlastWave2")

        bw1_integral_tot = bw1.Integral(0, 10, 1e-8)
        bw2_integral_tot = bw2.Integral(0, 10, 1e-8)

        bw_integral_tot = bw1_integral_tot + bw2_integral_to

        bw1_integral_range = bw1.Integral(pt_range[0], pt_range[1], 1e-8)
        bw2_integral_range = bw2.Integral(pt_range[0], pt_range[1], 1e-8)

        bw_integral_range = bw1_integral_range + bw2_integral_range

    else:
        if cent_class == 3:
            cent_class = 2

        bw = bw_file.Get('BlastWave/BlastWave{}'.format(cent_class))

        bw_integral_tot = bw.Integral(0, 10, 1e-8)
        bw_integral_range = bw.Integral(pt_range[0], pt_range[1], 1e-8)

    pt_width = pt_range[1] - pt_range[0]

    exp_yield = 2 * scale_factor[cent_class] * bw_integral_range / pt_width / bw_integral_tot

    return exp_yield


def expected_signal(n_ev, eff_presel, eff_bdt, pt_range, cent_class):
    signal_raw = expected_signal_raw(pt_range, cent_class)

    return int(round(n_ev * signal_raw * (pt_range[1] - pt_range[0]) * eff_presel * eff_bdt))


def significance_error(signal, background):
    signal_error = np.sqrt(signal +  1e-10)
    background_error = np.sqrt(background +  1e-10)

    sb = signal + background + 1e-10
    sb_sqrt = np.sqrt(sb)

    s_propag = (sb_sqrt + signal / (2 * sb_sqrt)) * signal_error
    b_propag = sb / (2 * sb_sqrt) * background_error

    return np.sqrt(s_propag * s_propag + b_propag * b_propag)


def expo(x, tau):
    return np.exp(-x / tau / 0.029979245800)
