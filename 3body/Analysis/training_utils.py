# this class has been created to generalize the training and to open the file.root just one time
# to achive that alse analysis_utils.py and Significance_Test.py has been modified

import os
import pickle

import numpy as np
import pandas as pd
import uproot
import xgboost as xgb
from scipy import stats
from scipy.stats import norm
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, train_test_split)

# ------------------------------
import analysis_utils as au
import Significance_Test as ST


class GeneralizedAnalysis:

    def __init__(self, mode, mc_file_name, data_file_name, cut_presel, bkg_selection):

        self.mode = mode
        self.cent_classes = [[0, 10], [10, 30], [30, 50], [50, 90]]
        self.n_events = [0, 0, 0, 0]

        self.df_signal = uproot.open(mc_file_name)['SignalTable'].pandas.df()
        self.df_generated = uproot.open(mc_file_name)['GenTable'].pandas.df()
        self.df_data = uproot.open(data_file_name)['DataTable'].pandas.df()

        self.df_data['Ct'] = self.df_data['DistOverP'] * 2.99131
        self.df_signal['Ct'] = self.df_signal['DistOverP'] * 2.99131

        self.df_signal['y'] = 1
        self.df_data['y'] = 0
        # dataframe for the background
        self.df_data = self.df_data.query(bkg_selection)
        # dataframe for the signal preselection cuts are applied
        self.df_signal_cuts = self.df_signaldfMCSig.query(cut_presel)

        utils_file_path = os.environ['HYPERML_UTILS']
        hist_centrality = uproot.open('{}/EventCounter.root'.format(utils_file_path))['fCentrality']

        for index in range(1, len(hist_centrality)):
            if index <= self.cent_classes[0][1]:
                self.n_events[0] = hist_centrality[index] + self.n_events[0]
            elif index <= self.cent_classes[1][1]:
                self.n_events[1] = hist_centrality[index] + self.n_events[1]
            elif index <= self.cent_classes[2][1]:
                self.n_events[2] = hist_centrality[index] + self.n_events[2]
            elif index <= self.cent_classes[3][1]:
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

        return len(self.df_signal_cuts.query(total_cut))/len(self.df_generated.query(total_cut_gen))

    def optimize_params_gs(self, dtrain, par):
        scoring = 'auc'
        early_stopping_rounds = 20
        num_rounds = 200
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
        gs_dict = {'first_par': {'name': 'max_depth', 'par_values': [i for i in range(2, 10, 2)]},
                   'second_par': {'name': 'min_child_weight', 'par_values': [i for i in range(0, 12, 2)]},
                   }
        par['max_depth'], par['min_child_weight'], _ = au.gs_2par(
            gs_dict, par, dtrain, num_rounds, 42, cv, scoring, early_stopping_rounds)

        gs_dict = {'first_par': {'name': 'subsample', 'par_values': [i/10. for i in range(4, 10)]},
                   'second_par': {'name': 'colsample_bytree', 'par_values': [i/10. for i in range(8, 10)]},
                   }
        par['subsample'], par['colsample_bytree'], _ = au.gs_2par(
            gs_dict, par, dtrain, num_rounds, 42, cv, scoring, early_stopping_rounds)
        gs_dict = {'first_par': {'name': 'gamma', 'par_values': [i/10. for i in range(0, 11)]}}
        par['gamma'], _ = au.gs_1par(gs_dict, par, dtrain, num_rounds, 42, cv, scoring, early_stopping_rounds)
        gs_dict = {'first_par': {'name': 'eta', 'par_values': [0.1, 0.05, 0.01, 0.005, 0.001]}}
        par['eta'], n = au.gs_1par(gs_dict, par, dtrain, num_rounds, 42, cv, scoring, early_stopping_rounds)
        return n

    def training_testing(
            self, training_columns, params_def, ct_cut=[0, 100],
            pt_cut=[2, 3],
            centrality_cut=[0, 10],
            num_rounds=200, draw=True, ROC=True, optimize=False):
        ct_min = ct_cut[0]
        ct_max = ct_cut[1]
        pt_min = pt_cut[0]
        pt_max = pt_cut[1]
        centrality_min = centrality_cut[0]
        centrality_max = centrality_cut[1]

        total_cut = '@ct_min<Ct<@ct_max and @pt_min<V0pt<@pt_max and @centrality_min<Centrality<@centrality_max'
        bkg = self.dfDataF.query(total_cut)
        sig = self.dfMCSigF.query(total_cut)
        print('condidates of bkg: ', len(bkg))
        print('condidates of sig: ', len(sig))
        if len(sig) is 0:
            print('no signal -> the model is not trained')
            return 0
        df = pd.concat([sig, bkg])
        traindata, testdata, ytrain, ytest = train_test_split(df[training_columns], df['y'], test_size=0.5)
        dtrain = xgb.DMatrix(data=np.asarray(traindata), label=ytrain, feature_names=training_columns)

        if optimize is True:
            num_rounds = self.optimize_params(dtrain, params_def)
            print(total_cut)
            print('num rounds: ', num_rounds)
            print('parameters: ', params_def)

        model = xgb.train(params_def, dtrain, num_boost_round=num_rounds)
        au.plot_output_train_test(
            model, traindata[training_columns],
            ytrain, testdata[training_columns],
            ytest, branch_names=training_columns, raw=True, log=True, draw=draw, ct_cut=ct_cut, pt_cut=pt_cut,
            centrality_cut=centrality_cut)
        droc = xgb.DMatrix(data=testdata)
        y_pred = model.predict(droc)
        if ROC is True:
            au.plot_roc(ytest, y_pred)
        self.traindata = traindata
        self.testdata = testdata
        self.ytrain = ytrain
        self.ytest = ytest
        return model

    def significance(self, model, training_columns, ct_cut=[0, 100], pt_cut=[2, 3], centrality_cut=[0, 10]):
        ct_min = ct_cut[0]
        ct_max = ct_cut[1]
        pt_max = pt_cut[1]
        pt_min = pt_cut[0]

        centrality_max = centrality_cut[1]
        centrality_min = centrality_cut[0]

        total_cut = '@ct_min<Ct<@ct_max and @pt_min<V0pt<@pt_max and @centrality_min<Centrality<@centrality_max'
        dtest = xgb.DMatrix(data=(self.testdata[training_columns]))
        self.testdata.eval('y = @self.ytest', inplace=True)
        y_pred = model.predict(dtest, output_margin=True)
        self.testdata.eval('Score = @y_pred', inplace=True)
        efficiency_array = au.EfficiencyVsCuts(self.testdata)
        i_cen = 0
        for index in range(0, len(self.Centrality)):
            if centrality_cut is self.Centrality[index]:
                i_cen = index
                break
        dfDataSig = self.dfData.query(total_cut)
        dtest = xgb.DMatrix(data=(dfDataSig[training_columns]))
        y_pred = model.predict(dtest, output_margin=True)
        dfDataSig.eval('Score = @y_pred', inplace=True)
        cut = ST.SignificanceScan(dfDataSig, ct_cut, pt_cut, centrality_cut, efficiency_array,
                                  self.EfficiencyPresel(ct_cut, pt_cut, centrality_cut), self.n_ev[i_cen])
        score_list = np.linspace(-3, 12.5, 100)
        for index in range(0, len(score_list)):
            if score_list[index] == cut:
                effBDT = efficiency_array[index]
        return (cut, effBDT)
