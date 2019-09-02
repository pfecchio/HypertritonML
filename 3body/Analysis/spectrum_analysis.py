import collections.abc
import os
import time
import warnings

import analysis_utils as au
import generalized_analysis as ga
import pandas as pd
import xgboost as xgb

import matplotlib.pyplot as plt

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

# paramaters of the xgboost regressor
XGBOOST_PARAMS = {
    # general parameters
    'silent': 1,  # print message (useful to understand what's happening)
    'nthread': 8,  # number of available threads
    # learning task parameters
    'objective': 'binary:logistic',
    'random_state': 42,
    'eval_metric': ['auc'],
    'tree_method': 'hist'
}

# range for the hyperparameters we want to maximize
HYPERPARAMS_RANGE = {
    # booster parameters
    'eta': (0.0001, 0.3),  # a kind of learning rate
    # defines the min sum of weights of all observations required in a child (regularization)
    'min_child_weight': (1, 12),
    'max_depth': (2, 20),  # defines the maximum depth of a single tree (regularization)
    'gamma': (0, 1.1),  # specifies the minimum loss reduction required to make a split
    'subsample': (0.3, 1.),  # denotes the fraction of observations to be randomly samples for each tree
    'colsample_bytree': (0.3, 0.95),  # denotes the fraction of columns to be randomly samples for each tree
    # 'lambda': (0,10),  # L2 regularization term on weights
    # 'alpha': (0,10),  # L1 regularization term on weight
    # should be used in case of high class imbalance as it helps in faster convergence
    'scale_pos_weight': (1., 10.)
}

# # features
TRAINING_COLUMNS = [
    'PtDeu', 'PtP', 'PtPi', 'nClsTPCDeu', 'nClsTPCP', 'nClsTPCPi', 'nClsITSDeu', 'nClsITSP', 'nClsITSPi',
    'nSigmaTPCDeu', 'nSigmaTPCP', 'nSigmaTPCPi', 'nSigmaTOFDeu', 'nSigmaTOFP', 'nSigmaTOFPi', 'DCA2xyPrimaryVtxDeu',
    'DCAxyPrimaryVtxP', 'DCAxyPrimaryVtxPi', 'DCAzPrimaryVtxDeu', 'DCAzPrimaryVtxP', 'DCAzPrimaryVtxPi',
    'DCAPrimaryVtxDeu', 'DCAPrimaryVtxP', 'DCAPrimaryVtxPi', 'DCAxyDecayVtxDeu', 'DCAxyDecayVtxP', 'DCAxyDecayVtxPi',
    'DCAzDecayVtxDeu', 'DCAzDecayVtxP', 'DCAzDecayVtxPi', 'DCADecayVtxDeu', 'DCADecayVtxP', 'DCADecayVtxPi',
    'TrackDistDeuP', 'TrackDistPPi', 'TrackDistDeuPi', 'CosPA']  # 38

# # features
# TRAINING_COLUMNS = [
#     'PtDeu', 'PtP', 'PtPi', 'nClsTPCDeu', 'nClsITSDeu', 'nClsITSPi', 'nSigmaTPCDeu', 'nSigmaTPCP', 'nSigmaTPCPi',
#     'trackChi2Deu', 'trackChi2P', 'DCA2xyPrimaryVtxDeu', 'DCAxyPrimaryVtxP', 'DCAxyPrimaryVtxPi', 'DCAzPrimaryVtxDeu',
#     'DCAzPrimaryVtxP', 'DCAzPrimaryVtxPi', 'DCAPrimaryVtxDeu', 'DCAPrimaryVtxP', 'DCAPrimaryVtxPi', 'DCAxyDecayVtxDeu',
#     'DCAxyDecayVtxP', 'DCAxyDecayVtxPi', 'DCAzDecayVtxDeu', 'DCAzDecayVtxP', 'DCAzDecayVtxPi', 'DCADecayVtxDeu',
#     'DCADecayVtxP', 'DCADecayVtxPi', 'TrackDistDeuP', 'TrackDistPPi', 'TrackDistDeuPi', 'CosPA']  # 33

TRAIN = True

table_path = os.environ['HYPERML_TABLES_3']
signal_table_path = '{}/HyperTritonTable_19d2.root'.format(table_path)
background_table_path = '{}/HyperTritonTable_18qr.root'.format(table_path)

bkg_selection = '(InvMass<2.98 or InvMass>3.005) and HypCandPt<=10'

analysis = ga.GeneralizedAnalysis(3, signal_table_path, background_table_path)

CENT_CLASS = [[0, 10], [10, 30], [30, 50], [50, 90]]
PT_BINS = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 9]]
# CT_BINS = [[0,2],[2,4],[4,6],[6,8],[8,10],[10,14],[14,18],[18,23],[23,28]]
CT_BINS = [0, 100]

# .txt file for writing the 98% eff BDT score
models_path = os.environ['HYPERML_MODELS_3']
eff_file = open('{}/eff98_score.txt'.format(models_path), 'w+')

# start timer for performance evaluation
start_time = time.time()

for cclass in CENT_CLASS:
    for ptbin in PT_BINS:
        print('============================================')
        print('centrality class: ', cclass)
        print('pT bin: ', ptbin)

        part_time = time.time()

        # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
        data = analysis.prepare_dataframe(
            TRAINING_COLUMNS, cclass, pt_range=ptbin, bkg_reduct=True, bkg_factor=10, test=True
            )

        # train and test the model with some performance plot
        model = analysis.train_test_model(
            data, TRAINING_COLUMNS, XGBOOST_PARAMS, hyperparams=HYPERPARAMS_RANGE, cent_class=cclass, pt_range=ptbin,
            optimize=False, num_rounds=500, es_rounds=20)

        print('--- model trained in {:.4f} minutes ---\n'.format((time.time() - part_time) / 60))

        analysis.save_model(model, ct_range=CT_BINS, cent_class=cclass, pt_range=ptbin)
        print('model saved\n')

        dtest = xgb.DMatrix(data=(data[2][TRAINING_COLUMNS]))

        y_pred = model.predict(dtest, output_margin=True)

        data[2].eval('Score = @y_pred', inplace=True)
        data[2].eval('y = @data[3]', inplace=True)

        efficiency, threshold = analysis.bdt_efficiency(
            data[2], pt_range=ptbin, cent_class=cclass, n_points=200)

        eff_tsd = tuple(zip(threshold, efficiency))

        # print on a file the score closest to efficiency 0.98
        for tsd, eff in reversed(eff_tsd):
            if eff > 0.98:
                score_eff_string = 'eff: {:>5.3f}    score: {:>8.5f}'.format(eff, tsd)
                eff_file.write(score_eff_string)
                print(score_eff_string)
                break

eff_file.close()
# print execution time to performance evaluation
print('')
print('--- {:.4f} minutes ---'.format((time.time() - start_time) / 60))


# model = analysis.train_test_model(
#     TRAINING_COLUMNS, XGBOOST_PARAMS, hyperparams=HYPERPARAMS_RANGE, cent_class=CENT_CLASS[1], pt_range=PT_BINS[1],
#     bkg_reduct=True, bkg_factor=10, test=False, optimize=True, num_rounds=500, es_rounds=20)
# analysis.save_model(model, cent_class=CENT_CLASS[1], pt_range=PT_BINS[1])

# cut_eff_dict['{}{}_{}{}_{}{}'.format(
#     CENT_CLASS[1][0],
#     CENT_CLASS[1][1],
#     PT_BINS[1][0],
#     PT_BINS[1][1],
#     CT_BINS[0],
#     CT_BINS[1])] = {'score_cut': score_cut, 'bdt_eff': bdt_efficiency}
