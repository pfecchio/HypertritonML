import collections.abc
import os
import time
import warnings

import analysis_utils as au
import generalized_analysis as ga
import pandas as pd
# import type
import xgboost as xgb

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

# paramaters of the xgboost regressor
XGBOOST_PARAMS = {
    # general parameters
    'silent': 0,  # print message (useful to understand what's happening)
    'nthread': 8,  # number of available threads
    # learning task parameters
    'objective': 'binary:logistic',
    'random_state': 42,
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
# TRAINING_COLUMNS = [
#     'HypCandPt', 'PtDeu', 'PtP', 'PtPi', 'nClsTPCDeu', 'nClsTPCP', 'nClsTPCPi', 'nClsITSDeu', 'nClsITSP',
#     'nClsITSPi', 'nSigmaTPCDeu', 'nSigmaTPCP', 'nSigmaTPCPi', 'nSigmaTOFDeu', 'nSigmaTOFP', 'nSigmaTOFPi',
#     'trackChi2Deu', 'trackChi2P', 'trackChi2Pi', 'vertexChi2', 'DCA2xyPrimaryVtxDeu', 'DCAxyPrimaryVtxP',
#     'DCAxyPrimaryVtxPi', 'DCAzPrimaryVtxDeu', 'DCAzPrimaryVtxP', 'DCAzPrimaryVtxPi', 'DCAPrimaryVtxDeu',
#     'DCAPrimaryVtxP', 'DCAPrimaryVtxPi', 'DCAxyDecayVtxDeu', 'DCAxyDecayVtxP', 'DCAxyDecayVtxPi', 'DCAzDecayVtxDeu',
#     'DCAzDecayVtxP', 'DCAzDecayVtxPi', 'DCADecayVtxDeu', 'DCADecayVtxP', 'DCADecayVtxPi', 'TrackDistDeuP',
#     'TrackDistPPi', 'TrackDistDeuPi', 'CosPA']  # 42

# features
TRAINING_COLUMNS = [
    'PtDeu', 'PtP', 'PtPi', 'nClsTPCDeu', 'nClsITSDeu', 'nClsITSPi', 'nSigmaTPCDeu', 'nSigmaTPCP', 'nSigmaTPCPi',
    'trackChi2Deu', 'trackChi2P', 'DCA2xyPrimaryVtxDeu', 'DCAxyPrimaryVtxP', 'DCAxyPrimaryVtxPi', 'DCAzPrimaryVtxDeu',
    'DCAzPrimaryVtxP', 'DCAzPrimaryVtxPi', 'DCAPrimaryVtxDeu', 'DCAPrimaryVtxP', 'DCAPrimaryVtxPi', 'DCAxyDecayVtxDeu',
    'DCAxyDecayVtxP', 'DCAxyDecayVtxPi', 'DCAzDecayVtxDeu', 'DCAzDecayVtxP', 'DCAzDecayVtxPi', 'DCADecayVtxDeu',
    'DCADecayVtxP', 'DCADecayVtxPi', 'TrackDistDeuP', 'TrackDistPPi', 'TrackDistDeuPi', 'CosPA']  # 33

table_path = os.environ['HYPERML_TABLES_3']
signal_table_path = '{}/HyperTritonTable_19d2.root'.format(table_path)
background_table_path = '{}/HyperTritonTable_18q.root'.format(table_path)

analysis = ga.GeneralizedAnalysis(3, signal_table_path, background_table_path)

CENT_BINS = [[0, 10], [10, 30], [30, 50], [50, 90]]
PT_BINS = [[1, 2], [2, 3], [3, 4], [4, 9]]

print('centrality class: ', CENT_BINS[1])
print('pT bin: ', PT_BINS[1])
print('')

# start timer for performance evaluation
start_time = time.time()

model = analysis.train_test(
    TRAINING_COLUMNS, XGBOOST_PARAMS, HYPERPARAMS_RANGE,
    cent_class=1,
    bkg_reduct=False, bkg_factor=10, draw=False, optimize=True)

# print execution time to performance evaluation
print('')
print('--- {:.2} minutes ---'.format((time.time() - start_time)/60))
