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

        self.df_data['Ct'] = self.df_data['DistOverP'] * 2.99131
        self.df_signal['Ct'] = self.df_signal['DistOverP'] * 2.99131

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

        total_cut = '{}<Ct<{} and {}<V0pt<{} and {}<Centrality<{}'.format(
            ct_min, ct_max, pt_min, pt_max, centrality_min, centrality_max)
        total_cut_gen = '{}<Ct<{} and {}<Pt<{} and {}<Centrality<{}'.format(
            ct_min, ct_max, pt_min, pt_max, centrality_min, centrality_max)

        return len(self.df_signal.query(total_cut))/len(self.df_generated.query(total_cut_gen))

    # def optimize_params_gs(self, dtrain, par):
    #     scoring = 'auc'
    #     early_stopping_rounds = 20
    #     num_rounds = 200

    #     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)

    #     gs_dict = {'first_par': {'name': 'max_depth', 'par_values': [i for i in range(2, 10, 2)]},
    #                'second_par': {'name': 'min_child_weight', 'par_values': [i for i in range(0, 12, 2)]},
    #                }
    #     par['max_depth'], par['min_child_weight'], _ = au.gs_2par(
    #         gs_dict, par, dtrain, num_rounds, 42, cv, scoring, early_stopping_rounds)

    #     gs_dict = {'first_par': {'name': 'subsample', 'par_values': [i/10. for i in range(4, 10)]},
    #                'second_par': {'name': 'colsample_bytree', 'par_values': [i/10. for i in range(8, 10)]},
    #                }
    #     par['subsample'], par['colsample_bytree'], _ = au.gs_2par(
    #         gs_dict, par, dtrain, num_rounds, 42, cv, scoring, early_stopping_rounds)
    #     gs_dict = {'first_par': {'name': 'gamma', 'par_values': [i/10. for i in range(0, 11)]}}
    #     par['gamma'], _ = au.gs_1par(gs_dict, par, dtrain, num_rounds, 42, cv, scoring, early_stopping_rounds)
    #     gs_dict = {'first_par': {'name': 'eta', 'par_values': [0.1, 0.05, 0.01, 0.005, 0.001]}}
    #     par['eta'], n = au.gs_1par(gs_dict, par, dtrain, num_rounds, 42, cv, scoring, early_stopping_rounds)
    #     return n

    def evaluate_hyperparams(
            self, dtrain, eta, min_child_weight, max_depth, gamma, subsample, colsample_bytree, scale_pos_weight,
            nfold=3):
        params = {'eval_metric': 'auc',
                  'eta': eta,
                  'min_child_weight': int(min_child_weight),
                  'max_depth': int(max_depth),
                  'gamma': gamma,
                  'subsample': subsample,
                  'colsample_bytree': colsample_bytree,
                  'scale_pos_weight': scale_pos_weight}
        # Used around 1000 boosting rounds in the full model
        cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=nfold)

        return cv_result['test-auc-mean'].iloc[-1]

    def optimize_params_bayes(self, dtrain, params, num_rounds=100, nfold=3, init_points=2, n_iter=5):
        # just an helper function
        def hyperparams_crossvalidation(
                eta, min_child_weight, max_depth, gamma, subsample, colsample_bytree, scale_pos_weight, nfold=3):
            return self.evaluate_hyperparams(
                dtrain, eta, min_child_weight, max_depth, gamma, subsample, colsample_bytree, scale_pos_weight, nfold=3)

        params_range = {'eta': tuple(params['eta'][1]),
                        'min_child_weight': tuple(params['min_child_weight'][1]),
                        'max_depth': tuple(params['max_depth'][1]),
                        'gamma': tuple(params['gamma'][1]),
                        'subsample': tuple(params['subsample'][1]),
                        'colsample_bytree': tuple(params['colsample_bytree'][1]),
                        # 'lambda' : tuple(params['lambda'][1]),
                        # 'alpha' : tuple(params['alpha'][1]),
                        'scale_pos_weight': tuple(params['scale_pos_weight'][1])}

        print(params_range)

        xgb_bo = BayesianOptimization(hyperparams_crossvalidation, {'eta': tuple(params['eta'][1]),
                                                                    'min_child_weight': tuple(params['min_child_weight'][1]),
                                                                    'max_depth': tuple(params['max_depth'][1]),
                                                                    'gamma': tuple(params['gamma'][1]),
                                                                    'subsample': tuple(params['subsample'][1]),
                                                                    'colsample_bytree': tuple(params['colsample_bytree'][1]),
                                                                    # 'lambda' : tuple(params['lambda'][1]),
                                                                    # 'alpha' : tuple(params['alpha'][1]),
                                                                    'scale_pos_weight': tuple(params['scale_pos_weight'][1])}, random_state=42)
        xgb_bo.maximize(init_points=init_points, n_iter=n_iter, acq='ei')

        max_params = xgb_bo.res['max']['max_params']
        return max_params

    def train_test(
            self, training_columns, params, ct_range=[0, 100],
            pt_range=[0, 100],
            cent_class=1, num_rounds=200, early_stopping_rounds=10, draw=True, ROC=True, optimize=False,
            optimize_mode='bayes', bkg_reduct=True, bkg_factor=1):
        ct_min = ct_range[0]
        ct_max = ct_range[1]
        pt_min = pt_range[0]
        pt_max = pt_range[1]

        cent_min = self.cent_class[cent_class][0]
        cent_max = self.cent_class[cent_class][1]

        if self.mode == 2:
            self.total_cut = '{}<Ct<{} and {}<V0pT<{} and {}<Centrality<{}'.format(
                ct_min, ct_max, pt_min, pt_max, cent_min, cent_max)
        if self.mode == 3:
            self.total_cut = '{}<Ct<{} and {}<HypCandPt<{} and {}<Centrality<{}'.format(
                ct_min, ct_max, pt_min, pt_max, cent_min, cent_max)

        bkg = self.df_data.query(self.total_cut)
        sig = self.df_signal.query(self.total_cut)

        test = True
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
        # define the xgb matrix objects
        dtrain = xgb.DMatrix(data=train_set, label=y_train, feature_names=training_columns)
        dtest = xgb.DMatrix(data=test_set)

        max_params = {}
        if optimize is True:
            if optimize_mode == 'bayes':
                max_params = self.optimize_params_bayes(dtrain, params,num_rounds=num_rounds)
            # if optimize_mode == 'gs'
            #     num_rounds = self.optimize_params_gs(dtrain, params)

        print('num rounds: ', num_rounds)
        print('parameters: ', max_params)
        print('')

        model = xgb.train(max_params, dtrain, num_boost_round=num_rounds, early_stopping_rounds=early_stopping_rounds)

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

    # def significance(self, model, training_columns, ct_cut=[0, 100], pt_cut=[2, 3], centrality_cut=[0, 10]):
    #     ct_min = ct_cut[0]
    #     ct_max = ct_cut[1]
    #     pt_max = pt_cut[1]
    #     pt_min = pt_cut[0]

    #     centrality_max = centrality_cut[1]
    #     centrality_min = centrality_cut[0]

    #     total_cut = '{}<Ct<{} and {}<V0pt<{} and {}<Centrality<{}'.format(
    #         ct_min, ct_max, pt_min, pt_max, centrality_min, centrality_max)

    #     dtest = xgb.DMatrix(data=(self.test_set[training_columns]))

    #     self.test_set.eval('y = {}'.format(self.y_test), inplace=True)

    #     y_pred = model.predict(dtest, output_margin=True)
    #     self.test_set.eval('Score = {}'.format(y_pred), inplace=True)

    #     efficiency_array = au.EfficiencyVsCuts(self.test_set)

    #     i_cen = 0
    #     for index in range(0, len(self.cent_classes)):
    #         if centrality_cut is self.cent_classes[index]:
    #             i_cen = index
    #             break
    #     dfDataSig = self.df_data.query(total_cut)
    #     dtest = xgb.DMatrix(data=(dfDataSig[training_columns]))
    #     y_pred = model.predict(dtest, output_margin=True)
    #     dfDataSig.eval('Score = @y_pred', inplace=True)
    #     cut = ST.SignificanceScan(dfDataSig, ct_cut, pt_cut, centrality_cut, efficiency_array,
    #                               self.preselection_efficiency(ct_cut, pt_cut, centrality_cut), self.n_events[i_cen])
    #     score_list = np.linspace(-3, 12.5, 100)
    #     for index in range(0, len(score_list)):
    #         if score_list[index] == cut:
    #             effBDT = efficiency_array[index]
    #     return (cut, effBDT)
