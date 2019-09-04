import collections.abc
import os
import time
import warnings

import analysis_utils as au
import generalized_analysis as ga
import pandas as pd
import xgboost as xgb
from generalized_analysis import GeneralizedAnalysis

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

TRAINING_COLUMNS = ['V0CosPA', 'ProngsDCA', 'PiProngPvDCAXY', 'He3ProngPvDCAXY',
                    'HypCandPt', 'He3ProngPvDCA', 'PiProngPvDCA', 'NpidClustersHe3', 'TPCnSigmaHe3']

# initialize the analysis
signal_table_path = os.environ['HYPERML_TABLES_2'] + '/SignalTable.root'
background_table_path = os.environ['HYPERML_TABLES_2'] + '/DataTable.root'

signal_selection = '2<=HypCandPt<=10'
backgound_selection = '(InvMass<2.98 or InvMass>3.005) and HypCandPt<=10'

analysis = GeneralizedAnalysis(2, signal_table_path, background_table_path,
                               signal_selection, backgound_selection, cent_class=[[0, 90]])

# ranges for the analysis
CENTRALITY_CLASS = [[0, 90]]
CT_BINS = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 14], [14, 18], [18, 23], [23, 28]]

TRAIN = True

# start timer for performance evaluation
start_time = time.time()

cclass = CENTRALITY_CLASS[0]

for ctbin in CT_BINS:
    print('============================================')
    print('centrality class: ', cclass)
    print('ct bin: ', ctbin)

    part_time = time.time()

    # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
    data = analysis.prepare_dataframe(TRAINING_COLUMNS, cclass, ct_range=ctbin)

    # train and test the model with some performance plot
    model = analysis.train_test_model(
        data, TRAINING_COLUMNS, XGBOOST_PARAMS, hyperparams=HYPERPARAMS_RANGE, ct_range=ctbin, cent_class=cclass,
        optimize=True, num_rounds=500, es_rounds=20)

    print('--- model trained in {:.4f} minutes ---\n'.format((time.time() - part_time) / 60))

    analysis.save_model(model, ct_range=CT_BINS, cent_class=cclass)
    print('model saved\n')

    dtest = xgb.DMatrix(data=(data[2][TRAINING_COLUMNS]))

    y_pred = model.predict(dtest, output_margin=True)

    data[2].eval('Score = @y_pred', inplace=True)
    data[2].eval('y = @data[3]', inplace=True)

    efficiency, threshold = analysis.bdt_efficiency(data[2], cent_class=cclass, ct_range=ctbin, n_points=200)

# print execution time to performance evaluation
print('')
print('--- {:.4f} minutes ---'.format((time.time() - start_time) / 60))
