#!/usr/bin/env python3
import argparse
import collections.abc
import os
import time
import warnings
from array import array

import numpy as np
import yaml

import analysis_utils as au
import pandas as pd
import plot_utils as pu
import xgboost as xgb
from generalized_analysis import GeneralizedAnalysis

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='Do the training', action='store_true')
parser.add_argument(
    '--test', help='Just test the functionalities (training with reduced number of candidates)', action='store_true')
parser.add_argument('-o', '--optimize', help='Run the optimization', action='store_true')
parser.add_argument('config', help='Path to the YAML configuration file')
args = parser.parse_args()

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# define paths for loading data and storing results
mc_path = os.path.expandvars(params['MC_PATH'])
data_path = os.path.expandvars(params['DATA_PATH'])

# start timer for performance evaluation
start_time = time.time()

signal_selection = '{}<=HypCandPt<={}'.format(params['PT_BINS'][0], params['PT_BINS'][-1])
backgound_selection = '{}<=HypCandPt<={}'.format(params['PT_BINS'][0], params['PT_BINS'][-1])

training_columns = params['TRAINING_COLUMNS']

analysis = GeneralizedAnalysis(
    3, mc_path, data_path, signal_selection, backgound_selection, cent_class=params['CENTRALITY_CLASS'],
    split='', dedicated_background=0, training_columns=training_columns)


for cclass in params['CENTRALITY_CLASS']:
    for ptbin in zip(params['PT_BINS'][:-1], params['PT_BINS'][1:]):
        for ctbin in zip(params['CT_BINS'][:-1], params['CT_BINS'][1:]):

            print('============================================')
            print('centrality: ', cclass, ' ct: ', ctbin, ' pT: ', ptbin, ' split: ', '')
            part_time = time.time()

            # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
            data = analysis.prepare_dataframe(
                training_columns,
                cclass, ct_range=ctbin, pt_range=ptbin, test=args.test, sig_nocent=False)
            model = analysis.load_model(ct_range=ctbin, cent_class=cclass, pt_range=ptbin, split_string='')

            # plot features importance
            pu.plot_feature_imp(
                data[2][training_columns],
                data[3],
                model, 3, ct_range=ctbin, pt_range=ptbin, cent_class=cclass,
                split_string='')

        print('--- features importance in {:.4f} minutes ---\n'.format((time.time() - part_time) / 60))

# print execution time to performance evaluation
print('')
print('--- {:.4f} minutes ---'.format((time.time() - start_time) / 60))
