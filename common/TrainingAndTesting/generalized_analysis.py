# this class has been created to generalize the training and to open the file.root just one time
# to achive that alse analysis_utils.py and Significance_Test.py has been modified

import os
import pickle

import numpy as np
import pandas as pd
import uproot
import xgboost as xgb
from bayes_opt import BayesianOptimization
from scipy import stats
from scipy.stats import norm
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, auc
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, train_test_split)

import analysis_utils as au


class GeneralizedAnalysis:

    def __init__(self, mode, mc_file_name, data_file_name, cut_presel=0, bkg_selection=0):

        self.mode = mode
        self.cent_class = [[0, 10], [10, 30], [30, 50], [50, 90]]
        self.n_events = [0, 0, 0, 0]

        if mode == 3:
            self.df_signal = uproot.open(mc_file_name)['SignalTable'].pandas.df()
            self.df_generated = uproot.open(mc_file_name)['GenerateTable'].pandas.df()
            self.df_data = uproot.open(data_file_name)['BackgroundTable'].pandas.df()

        if mode == 2:
            self.df_signal = uproot.open(mc_file_name)['SignalTable'].pandas.df()
            self.df_generated = uproot.open(mc_file_name)['GenTable'].pandas.df()
            self.df_data = uproot.open(data_file_name)['DataTable'].pandas.df()

        self.df_data['ct'] = self.df_data['DistOverP'] * 2.99131
        self.df_signal['ct'] = self.df_signal['DistOverP'] * 2.99131

        self.df_signal['y'] = 1
        self.df_data['y'] = 0

        # dataframe for signal and background with preselection
        if not cut_presel == 0:
            self.df_data = self.df_data.query(bkg_selection)
        if not bkg_selection == 0:
            self.df_signal = self.df_signal.query(cut_presel)

        utils_file_path = os.environ['HYPERML_UTILS']
        hist_centrality = uproot.open('{}/EventCounter.root'.format(utils_file_path))['fCentrality']

        for index in range(1, len(hist_centrality)):
            if index <= self.cent_class[0][1]:
                self.n_events[0] = hist_centrality[index] + self.n_events[0]
            elif index <= self.cent_class[1][1]:
                self.n_events[1] = hist_centrality[index] + self.n_events[1]
            elif index <= self.cent_class[2][1]:
                self.n_events[2] = hist_centrality[index] + self.n_events[2]
            elif index <= self.cent_class[3][1]:
                self.n_events[3] = hist_centrality[index] + self.n_events[3]

    # function to compute the preselection cuts efficiency
    def preselection_efficiency(self, ct_cut=[0, 100], pt_cut=[0, 12], centrality_cut=[0, 100]):
        ct_min = ct_cut[0]
        ct_max = ct_cut[1]
        pt_max = pt_cut[1]
        pt_min = pt_cut[0]

        centrality_max = centrality_cut[1]
        centrality_min = centrality_cut[0]

        total_cut = '{}<ct<{} and {}<V0pt<{} and {}<Centrality<{}'.format(
            ct_min, ct_max, pt_min, pt_max, centrality_min, centrality_max)
        total_cut_gen = '{}<ct<{} and {}<Pt<{} and {}<Centrality<{}'.format(
            ct_min, ct_max, pt_min, pt_max, centrality_min, centrality_max)

        return len(self.df_signal.query(total_cut))/len(self.df_generated.query(total_cut_gen))

    # target function for the bayesian hyperparameter optimization
    def evaluate_hyperparams(
            self, dtrain, reg_params, eta, min_child_weight, max_depth, gamma, subsample, colsample_bytree,
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
        best_score = 100 * (cv_result['test-auc-mean'].max() - 0.99)

        round_score_list.append(tuple([best_score, best_boost_rounds]))

        return best_score

    # function that manage the bayesian optimization
    def optimize_params_bayes(
            self, dtrain, reg_params, hyperparams, num_rounds=100, es_rounds=2, nfold=3, init_points=1, n_iter=1):
        round_score_list = []

        # just an helper function
        def hyperparams_crossvalidation(
                eta, min_child_weight, max_depth, gamma, subsample, colsample_bytree, scale_pos_weight):
            return self.evaluate_hyperparams(
                dtrain, reg_params, eta, min_child_weight, max_depth, gamma, subsample, colsample_bytree,
                scale_pos_weight, num_rounds, es_rounds, nfold, round_score_list)

        print('')

        optimizer = BayesianOptimization(f=hyperparams_crossvalidation,
                                         pbounds=hyperparams, verbose=2, random_state=42)
        optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='poi')
        print('')

        best_numrounds = max(round_score_list, key=lambda x: x[0])[1]

        # extract and show the results of the optimization
        max_params = {
            'eta': optimizer.max['params']['eta'],
            'min_child_weight': int(optimizer.max['params']['min_child_weight']),
            'max_depth': int(optimizer.max['params']['max_depth']),
            'gamma': optimizer.max['params']['gamma'],
            'subsample': optimizer.max['params']['subsample'],
            'colsample_bytree': optimizer.max['params']['colsample_bytree'],
            # 'lambda': optimizer.max['params']['lambda'],
            # 'alpha': optimizer.max['params']['alpha'],
            'scale_pos_weight': optimizer.max['params']['scale_pos_weight'],
        }
        print('Best target: {0:.6f}'.format(optimizer.max['target']))
        print('Best parameters: {}'.format(max_params))
        print('Best num_rounds: {}\n'.format(best_numrounds))

        return max_params, best_numrounds

    # manage all the training stuffs
    def train_test(
            self, training_columns, reg_params, hyperparams=0, ct_range=[0, 100],
            pt_range=[0, 100],
            cent_class=1, num_rounds=1000, es_rounds=20, draw=True, ROC=True, optimize=False, optimize_mode='bayes',
            bkg_reduct=True, bkg_factor=1):
        ct_min = ct_range[0]
        ct_max = ct_range[1]
        pt_min = pt_range[0]
        pt_max = pt_range[1]

        cent_min = self.cent_class[cent_class][0]
        cent_max = self.cent_class[cent_class][1]

        if self.mode == 2:
            self.total_cut = '{}<ct<{} and {}<V0pT<{} and {}<Centrality<{}'.format(
                ct_min, ct_max, pt_min, pt_max, cent_min, cent_max)
        if self.mode == 3:
            self.total_cut = '{}<ct<{} and {}<HypCandPt<{} and {}<Centrality<{}'.format(
                ct_min, ct_max, pt_min, pt_max, cent_min, cent_max)

        bkg = self.df_data.query(self.total_cut)
        sig = self.df_signal.query(self.total_cut)

        test = False
        if test:
            sig = sig.sample(n=1000)
            bkg = bkg.sample(n=10000)

        if bkg_reduct:
            n_bkg = len(sig) * bkg_factor
            if n_bkg < len(bkg):
                bkg = bkg.sample(n=n_bkg)

        print('number of background candidates: ', len(bkg))
        print('number of signal candidates: ', len(sig))
        print('')

        df = pd.concat([sig, bkg])
        train_set, test_set, y_train, y_test = train_test_split(df[training_columns], df['y'], test_size=0.5)

        dtrain = xgb.DMatrix(data=train_set, label=y_train, feature_names=training_columns)
        dtest = xgb.DMatrix(data=test_set, label=y_test, feature_names=training_columns)

        # manage the optimization process
        if optimize is True:
            if optimize_mode == 'bayes':
                max_params, best_numrounds = self.optimize_params_bayes(
                    dtrain, reg_params, hyperparams, num_rounds=num_rounds, es_rounds=es_rounds, init_points=10,n_iter=100)
            # if optimize_mode == 'gs'
            #     num_rounds = self.optimize_params_gs(dtrain, params)
        else:   # manage the default params
            if hyperparams == 0:
                max_params = reg_params
                best_numrounds = num_rounds

        # join the dictionaries of the regressor params with the maximized hyperparams
        max_params = {**max_params, **reg_params}

        # get the best num_rounds

        model = xgb.train(max_params, dtrain, num_boost_round=best_numrounds)

        au.plot_output_train_test(
            model, train_set[training_columns],
            y_train, test_set[training_columns],
            y_test, branch_names=training_columns, raw=True, log=True, draw=draw, ct_range=ct_range, pt_range=pt_range,
            cent_range=[cent_min, cent_max], mode=self.mode)

        y_pred = model.predict(dtest)
        roc_score = roc_auc_score(y_test, y_pred)

        print('roc_auc_score: {}'.format(roc_score))

        # if ROC is True:
        #     au.plot_roc(y_test, y_pred)

        self.train_set = train_set
        self.test_set = test_set
        self.y_train = y_train
        self.y_test = y_test

        return model
