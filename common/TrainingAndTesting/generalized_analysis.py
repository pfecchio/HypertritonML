# this class has been created to generalize the training and to open the file.root just one time
# to achive that alse analysis_utils.py and Significance_Test.py has been modified

import csv
import os

import numpy as np

import analysis_utils as au
import pandas as pd
import plot_utils as pu
import uproot
import xgboost as xgb
from bayes_opt import BayesianOptimization
from ROOT import TF1, TFile, gDirectory
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


class GeneralizedAnalysis:

    def __init__(
            self, mode, mc_file_name, data_file_name, sig_selection=0, bkg_selection=0,
            cent_class=[[0, 10],
                        [10, 30],
                        [30, 50],
                        [50, 90]], split=0, dedicated_background=0, training_columns=[]):
        self.mode = mode

        self.cent_class = cent_class.copy()
        self.n_events = [0, 0, 0, 0]

        if mode == 3:
            self.df_signal = uproot.open(mc_file_name)['SignalTable'].pandas.df()
            self.df_generated = uproot.open(mc_file_name)['GenerateTable'].pandas.df()
            self.df_data = uproot.open(data_file_name)['BackgroundTable'].pandas.df()

        if mode == 2:
            self.df_signal = uproot.open(mc_file_name)['SignalTable'].pandas.df()
            self.df_generated = uproot.open(mc_file_name)['GenTable'].pandas.df()
            self.df_data = uproot.open(data_file_name)['DataTable'].pandas.df()

        self.df_signal['y'] = 1
        self.df_data['y'] = 0

        self.df_data_bkg = self.df_data.sample(
            n=round(len(self.df_data) * dedicated_background)) if dedicated_background != 0 else self.df_data

        # backup of the data without any selections for the significance scan
        self.df_data_all = self.df_data.drop(self.df_data_bkg.index) if dedicated_background != 0 else self.df_data

        if split == 'antimatter':
            self.df_data_all = self.df_data_all.query('ArmenterosAlpha < 0')
            self.df_generated = self.df_generated.query('matter < 0.5')
        if split == 'matter':
            self.df_data_all = self.df_data_all.query('ArmenterosAlpha > 0')
            self.df_generated = self.df_generated.query('matter > 0.5')

        # dataframe for signal and background with preselection
        if isinstance(sig_selection, str):
            self.df_signal = self.df_signal.query(sig_selection)
        if isinstance(bkg_selection, str):
            self.df_data = self.df_data.query(bkg_selection)
            self.df_data_bkg = self.df_data_bkg.query(bkg_selection)
        
        # df = pd.concat([self.df_signal, self.df_data_bkg])

        # columns = training_columns.copy()
        # columns.append('InvMass')

        # pu.plot_distr(df, column=training_columns, mode=self.mode)
        # pu.plot_corr(df, columns, mode=self.mode)

        # del df

        if mode == 2:
            self.hist_centrality = uproot.open(data_file_name)['EventCounter']
            self.n_events = []
            for cent in self.cent_class:
                self.n_events.append(sum(self.hist_centrality[cent[0]+1:cent[1]]))

    # function to prepare the dataframe for the training and testing
    def prepare_dataframe(
            self, training_columns, cent_class, pt_range=[0, 10],
            ct_range=[0, 100],
            test=False, bkg_reduct=False, bkg_factor=1, sig_nocent=False):
        data_range_bkg = '{}<ct<{} and {}<HypCandPt<{} and {}<centrality<{}'.format(
            ct_range[0], ct_range[1], pt_range[0], pt_range[1], cent_class[0], cent_class[1])

        columns = training_columns.copy()
        columns.append('InvMass')

        if 'HypCandPt' not in columns:
            columns.append('HypCandPt')
        if 'ct' not in columns:
            columns.append('ct')
        
        if sig_nocent:
            data_range_sig = '{}<ct<{} and {}<HypCandPt<{}'.format(
                ct_range[0], ct_range[1], pt_range[0], pt_range[1])
        else:
            data_range_sig = data_range_bkg


        bkg = self.df_data_bkg.query(data_range_bkg)
        sig = self.df_signal.query(data_range_sig)

        if test:
            if len(sig) >= 1000:
                sig = sig.sample(n=1000)
            if len(bkg) >= 1000:
                bkg = bkg.sample(n=1000)

            self.df_data=self.df_data.sample(n=1000)    

        if bkg_reduct:
            n_bkg = int(len(bkg) * bkg_factor)
            if n_bkg < len(bkg):
                bkg = bkg.sample(n=n_bkg)

        print('\nnumber of background candidates: {}'.format(len(bkg)))
        print('number of signal candidates: {}\n'.format(len(sig)))

        df = pd.concat([sig, bkg])
        train_set, test_set, y_train, y_test = train_test_split(df[columns], df['y'], test_size=0.5, random_state=42)

        return train_set, y_train, test_set, y_test

    # function to compute the preselection cuts efficiency

    def preselection_efficiency(self, ct_range=[0, 100], pt_range=[0, 10], cent_class=[0, 100]):
        ct_min = ct_range[0]
        ct_max = ct_range[1]

        pt_min = pt_range[0]
        pt_max = pt_range[1]

        cent_min = cent_class[0]
        cent_max = cent_class[1]

        total_cut = '{}<ct<{} and {}<HypCandPt<{} and {}<centrality<{}'.format(
            ct_min, ct_max, pt_min, pt_max, cent_min, cent_max)
        total_cut_gen = '{}<ct<{} and {}<pT<{} and {}<centrality<{} and abs(rapidity)<0.5'.format(
            ct_min, ct_max, pt_min, pt_max, cent_min, cent_max)

        eff = len(self.df_signal.query(total_cut))/len(self.df_generated.query(total_cut_gen))

        return eff

    # function that manage the bayesian optimization
    def optimize_params_bayes(
            self, data, training_columns, reg_params, hyperparams, nfold=3, init_points=5,
            n_iter=5):

        # just an helper function
        def hyperparams_crossvalidation(
                max_depth, learning_rate, n_estimators, gamma, min_child_weight, subsample, colsample_bytree):
            return au.evaluate_hyperparams(
                data, training_columns, reg_params, max_depth, learning_rate, n_estimators, gamma, min_child_weight,
                subsample, colsample_bytree, nfold)

        print('')

        optimizer = BayesianOptimization(f=hyperparams_crossvalidation,
                                         pbounds=hyperparams, verbose=2, random_state=42)
        optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='poi')
        print('')

        # extract and show the results of the optimization
        max_params = {
            'max_depth': int(optimizer.max['params']['max_depth']),
            'learning_rate': optimizer.max['params']['learning_rate'],
            'n_estimators': int(optimizer.max['params']['n_estimators']),
            'gamma': optimizer.max['params']['gamma'],
            'min_child_weight': int(optimizer.max['params']['min_child_weight']),
            'subsample': optimizer.max['params']['subsample'],
            'colsample_bytree': optimizer.max['params']['colsample_bytree'],
            # 'lambda': optimizer.max['params']['lambda'],
            # 'alpha': optimizer.max['params']['alpha'],
        }
        print('Best target: {0:.6f}'.format(optimizer.max['target']))
        print('Best parameters: {}'.format(max_params))

        return max_params

    # function that manage the grid search
    def optimize_params_gs(self, dtrain, params):
        num_rounds = 500
        early_stopping_rounds = 50

        scoring = 'auc'

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        gs_dict = {'first_par': {'name': 'max_depth', 'par_values': [i for i in range(2, 10, 2)]}, 'second_par': {
            'name': 'min_child_weight', 'par_values': [i for i in range(0, 12, 2)]}, }

        params['max_depth'], params['min_child_weight'], _ = au.gs_2par(
            gs_dict, params, dtrain, num_rounds, 42, folds, scoring, early_stopping_rounds)

        gs_dict = {
            'first_par': {'name': 'subsample', 'par_values': [i / 10. for i in range(4, 10)]},
            'second_par': {'name': 'colsample_bytree', 'par_values': [i / 10. for i in range(8, 10)]}, }

        params['subsample'], params['colsample_bytree'], _ = au.gs_2par(
            gs_dict, params, dtrain, num_rounds, 42, folds, scoring, early_stopping_rounds)

        gs_dict = {'first_par': {'name': 'gamma', 'par_values': [i / 10. for i in range(0, 11)]}}
        params['gamma'], _ = au.gs_1par(gs_dict, params, dtrain, num_rounds, 42, folds, scoring, early_stopping_rounds)

        gs_dict = {'first_par': {'name': 'eta', 'par_values': [0.1, 0.05, 0.01, 0.005, 0.001]}}
        params['eta'], n = au.gs_1par(gs_dict, params, dtrain, num_rounds, 42, folds, scoring, early_stopping_rounds)

        return params, n

    # manage all the training stuffs
    def train_test_model(
            self, data, training_columns, reg_params, ct_range=[0, 100],
            pt_range=[0, 10],
            cent_class=[0, 10],
            hyperparams=0, optimize=False, optimize_mode='bayes', split_string=''):
        data_train = [data[0], data[1]]

        # manage the optimization process
        if optimize:
            if optimize_mode == 'bayes':
                print('Hyperparameters optimization: ...', end='\r')
                max_params = self.optimize_params_bayes(
                    data_train, training_columns, reg_params, hyperparams, init_points=10, n_iter=10)
                print('Hyperparameters optimization: Done!\n')
            if optimize_mode == 'gs':
                print('Hyperparameters optimization: ...', end='\r')
                max_params, best_numrounds = self.optimize_params_gs(data_train, hyperparams)
                print('Hyperparameters optimization: Done!\n')
        else:  # manage the default params
            max_params = hyperparams

        # join the dictionaries of the regressor params with the maximized hyperparams
        best_params = {**max_params, **reg_params}


        # final training with the optimized hyperparams
        print('Training the final model: ...', end='\r')
        model = xgb.XGBClassifier(**best_params)
        model.fit(data[0][training_columns], data[1])
        print('{}'.format(model.get_params()))
        print('Training the final model: Done!\n')

        # BDT output distributions plot
        fig_path = os.environ['HYPERML_FIGURES_{}'.format(self.mode)] + '/TrainTest'
        pu.plot_output_train_test(
            model, data[0][training_columns],
            data[1], data[2][training_columns],
            data[3], features=training_columns, raw=True, log=True, ct_range=ct_range, pt_range=pt_range,
            cent_class=cent_class, path=fig_path, mode=self.mode, split_string=split_string)

        pu.plot_feature_imp(data[0][training_columns], data[1] ,model, self.mode, ct_range=ct_range, pt_range=pt_range,
            cent_class=cent_class, split_string=split_string)    
        
        # test the model performances
        print('Testing the model: ...', end='\r')
        y_pred = model.predict(data[2][training_columns])
        roc_score = roc_auc_score(data[3], y_pred)
        print('Testing the model: Done!\n')

        print('ROC_AUC_score: {}\n'.format(roc_score))
        print('==============================\n')

        return model

    def save_model(self, model, cent_class=[0, 90], pt_range=[0, 10], ct_range=[0, 100], split_string=''):
        models_path = os.environ['HYPERML_MODELS_{}'.format(self.mode)]
        filename = '/BDT_{}{}_{}{}_{}{}{}.model'.format(cent_class[0],
                                                    cent_class[1],
                                                    pt_range[0],
                                                    pt_range[1],
                                                    ct_range[0],
                                                    ct_range[1], 
                                                    split_string)

        model.save_model(models_path + filename)
        print('Model saved.\n')

    def load_model(self, cent_class=[0, 90], pt_range=[0, 10], ct_range=[0, 100], split_string=''):
        models_path = os.environ['HYPERML_MODELS_{}'.format(self.mode)]
        filename = '/BDT_{}{}_{}{}_{}{}{}.model'.format(cent_class[0],
                                                    cent_class[1],
                                                    pt_range[0],
                                                    pt_range[1],
                                                    ct_range[0],
                                                    ct_range[1], 
                                                    split_string)

        model = xgb.XGBClassifier()
        model.load_model(models_path + filename)
        print('Model loaded.\n')
        return model

    def bdt_efficiency_array(self, df, ct_range=[0, 100], pt_range=[0, 10], cent_class=[0, 90], n_points=10, split_string=''):
        min_score = df['Score'].min()
        max_score = df['Score'].max()

        threshold = np.linspace(min_score, max_score, n_points)

        efficiency = []

        n_sig = sum(df['y'])

        for t in threshold:
            df_selected = df.query('Score>@t')['y']
            sig_selected = np.sum(df_selected)
            efficiency.append(sig_selected / n_sig)

        pu.plot_bdt_eff(threshold, efficiency, self.mode, ct_range, pt_range, cent_class, split_string=split_string)

        return efficiency, threshold

    def bdt_efficiency(self, df, cut):
        n_sig = sum(df['y'])
        df_selected = df.query('Score>@cut')['y']
        sig_selected = np.sum(df_selected)

        return sig_selected / n_sig

    def score_from_efficiency(self, model, test_data, efficiency_cut, training_columns, ct_range, pt_range, cent_class, split_string=''):
        y_pred = model.predict(test_data[0][training_columns], output_margin=True)

        test_data[0].eval('Score = @y_pred', inplace=True)
        test_data[0].eval('y = @test_data[1]', inplace=True)

        bdt_efficiency, threshold_space = self.bdt_efficiency_array(
            test_data[0], ct_range, pt_range, cent_class, n_points=1000, split_string = split_string)

        for index in range(0, len(bdt_efficiency)):
            bdt_efficiency[index] = round(bdt_efficiency[index], 2)

        threshold_cut = []
        index = -1
        for cut in efficiency_cut:
            done = False
            for (bdt, threshold) in zip(bdt_efficiency, threshold_space):
                if cut == bdt:
                    if not done:
                        threshold_cut.append(threshold)
                        index = index + 1
                        done = True
                    else:
                        threshold_cut[index] = threshold
            if not done:
                threshold_cut.append(20)

        return [[a, b] for a, b in zip(threshold_cut, efficiency_cut)]

    def significance_scan(
            self, test_data, model, training_columns, ct_range=[0, 100],
            pt_range=[0, 10],
            cent_class=[0, 100], custom=True, n_points=100, split_string=''):
        print('Significance scan: ...', end='\r')

        ct_min = ct_range[0]
        ct_max = ct_range[1]

        pt_min = pt_range[0]
        pt_max = pt_range[1]

        cent_min = cent_class[0]
        cent_max = cent_class[1]

        data_range = '{}<ct<{} and {}<HypCandPt<{} and {}<centrality<{}'.format(
            ct_range[0], ct_range[1], pt_range[0], pt_range[1], cent_class[0], cent_class[1])
        data_range_array = [ct_min, ct_max, pt_min, pt_max, cent_min, cent_max]

        columns = training_columns.copy()
        columns.append('InvMass')
        df_bkg = self.df_data.query(data_range)[columns]
        y_pred = model.predict(test_data[0][training_columns], output_margin=True)
        y_pred_bkg = model.predict(df_bkg[training_columns], output_margin=True)

        test_data[0].eval('Score = @y_pred', inplace=True)
        test_data[0].eval('y = @test_data[1]', inplace=True)
        df_bkg.eval('Score = @y_pred_bkg', inplace=True)

        bdt_efficiency, threshold_space = self.bdt_efficiency_array(
            test_data[0], ct_range, pt_range, cent_class, n_points, split_string=split_string)

        expected_signal = []
        significance = []
        significance_error = []
        significance_custom = []
        significance_custom_error = []

        hyp_lifetime = 206

        bw_file = TFile(os.environ['HYPERML_UTILS'] + '/BlastWaveFits.root')
        bw = [bw_file.Get("BlastWave/BlastWave{}".format(i)) for i in [0, 1, 2]]

        for index, tsd in enumerate(threshold_space):
            df_selected = df_bkg.query('Score>@tsd')

            counts, bins = np.histogram(df_selected['InvMass'], bins=45, range=[2.96, 3.05])
            bin_centers = 0.5 * (bins[1:] + bins[:-1])

            side_map = (bin_centers < 2.98) + (bin_centers > 3.005)
            mass_map = np.logical_not(side_map)
            bins_side = bin_centers[side_map]
            counts_side = counts[side_map]

            h, residuals, _, _, _ = np.polyfit(bins_side, counts_side, 2, full=True)
            y = np.polyval(h, bins_side)

            eff_presel = self.preselection_efficiency(ct_range, pt_range, cent_class)

            exp_signal_ctint = au.expected_signal_counts(
                bw, pt_range, eff_presel * bdt_efficiency[index],
                cent_class, self.hist_centrality)

            if split_string is not '':
                exp_signal_ctint = 0.5 * exp_signal_ctint

            ctrange_correction = au.expo(ct_min, hyp_lifetime)-au.expo(ct_max, hyp_lifetime)

            exp_signal = exp_signal_ctint * ctrange_correction
            exp_background = sum(np.polyval(h, bin_centers[mass_map]))

            expected_signal.append(exp_signal)

            if (exp_background < 0):
                exp_background = 0

            sig = exp_signal / np.sqrt(exp_signal + exp_background + 1e-10)
            sig_error = au.significance_error(exp_signal, exp_background)

            significance.append(sig)
            significance_error.append(sig_error)

            sig_custom = sig * bdt_efficiency[index]
            sig_custom_error = sig_error * bdt_efficiency[index]

            significance_custom.append(sig_custom)
            significance_custom_error.append(sig_custom_error)

        # pu.plot_efficiency_significance(self.mode, threshold_space, significance, bdt_efficiency, data_range_array)

        nevents = sum(self.hist_centrality[cent_class[0]+1:cent_class[1]])

        if custom:
            max_index = np.argmax(significance_custom)
            max_score = threshold_space[max_index]
            max_significance = significance_custom[max_index]

            pu.plot_significance_scan(
                max_index, significance_custom, significance_custom_error, expected_signal, df_bkg,
                threshold_space, data_range_array, bin_centers, nevents, self.mode, custom=True, split_string=split_string)

        else:
            max_index = np.argmax(significance)
            max_score = threshold_space[max_index]
            max_significance = significance[max_index]

            pu.plot_significance_scan(
                max_index, significance, significance_error, expected_signal, df_bkg, threshold_space,
                data_range_array, bin_centers, nevents, self.mode, custom=False, split_string=split_string)

        bdt_eff_max_score = bdt_efficiency[max_index]

        print('Significance scan: Done!\n')

        # return max_score, bdt_eff_max_score, max_significance
        return max_score, bdt_eff_max_score
