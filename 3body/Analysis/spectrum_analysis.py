import os
import time

import pandas as pd
import xgboost as xgb

import analysis_utils as au
import generalized_analysis as ga

# xgb hyperparameters that we are going to tune with bayesia optimization
params_range = {
    # general parameters
    'silent': 0,  # print message (useful to understand what's happening)
    'nthread': 4,  # number of available threads
    # booster parameters
    'eta': [0.05, [0.0001, 0.3]],  # a kind of learning rate
    # defines the min sum of weights of all observations required in a child (regularization)
    'min_child_weight': [5, [1, 12]],
    'max_depth': [8, [2, 20]],  # defines the maximum depth of a single tree (regularization)
    'gamma': [0.7, [0, 1.1]],  # specifies the minimum loss reduction required to make a split
    'subsample': [0.8, [0.3, 1.5]],  # denotes the fraction of observations to be randomly samples for each tree
    'colsample_bytree': [0.1, [0.3,10.]],  # denotes the fraction of columns to be randomly samples for each tree
    # 'lambda': [0, [0,10]],  # L2 regularization term on weights
    # 'alpha': [0, [0,10]],  # L1 regularization term on weight
    'scale_pos_weight': [1., [1.,10.]],  # should be used in case of high class imbalance as it helps in faster convergence
    # learning task parameters
    'objective': 'binary:logistic',
    'random_state': 42,
    'tree_method': 'hist',
}

default_params = {
    # general parameters
    'silent': 0,  # print message (useful to understand what's happening)
    'nthread': 4,  # number of available threads
    # booster parameters
    'eta': 0.05,
    'min_child_weight': 5,
    'max_depth': 8,
    'gamma': 0.7,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'scale_pos_weight': 1.,
    'objective': 'binary:logistic',
    'random_state': 42,
    'tree_method': 'hist',
}

training_columns = [
    'HypCandPt', 'PtDeu', 'PtP', 'PtPi', 'nClsTPCDeu', 'nClsTPCP', 'nClsTPCPi', 'nClsITSDeu', 'nClsITSP',
    'nClsITSPi', 'nSigmaTPCDeu', 'nSigmaTPCP', 'nSigmaTPCPi', 'nSigmaTOFDeu', 'nSigmaTOFP', 'nSigmaTOFPi',
    'trackChi2Deu', 'trackChi2P', 'trackChi2Pi', 'vertexChi2', 'DCA2xyPrimaryVtxDeu', 'DCAxyPrimaryVtxP',
    'DCAxyPrimaryVtxPi', 'DCAzPrimaryVtxDeu', 'DCAzPrimaryVtxP', 'DCAzPrimaryVtxPi', 'DCAPrimaryVtxDeu',
    'DCAPrimaryVtxP', 'DCAPrimaryVtxPi', 'DCAxyDecayVtxDeu', 'DCAxyDecayVtxP', 'DCAxyDecayVtxPi', 'DCAzDecayVtxDeu',
    'DCAzDecayVtxP', 'DCAzDecayVtxPi', 'DCADecayVtxDeu', 'DCADecayVtxP', 'DCADecayVtxPi', 'TrackDistDeuP',
    'TrackDistPPi', 'TrackDistDeuPi', 'CosPA']  # 42

table_path = os.environ['HYPERML_TABLES_3']
signal_table_path = '{}/HyperTritonTable_19d2.root'.format(table_path)
background_table_path = '{}/HyperTritonTable_18q.root'.format(table_path)

analysis = ga.GeneralizedAnalysis(3, signal_table_path, background_table_path)

cent_bins = [[0, 10], [10, 30], [30, 50], [50, 90]]
pt_bins = [[1, 2], [2, 3], [3, 4], [4, 9]]

cut_saved = []
eff_BDT = []

print('centrality class: ', cent_bins[1])
print('pT bin: ', pt_bins[1])
print('')

# start timer for performance evaluation
start_time = time.time()

model = analysis.train_test(
    training_columns, params_range,
    cent_class=1,
    bkg_reduct=False, bkg_factor=10, draw=False, optimize=True)

# print execution time to performance evaluation
print('')
print('--- {} minutes ---'.format((time.time() - start_time)/60))
