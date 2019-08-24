import io
from contextlib import redirect_stdout

import xgboost as xgb


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
