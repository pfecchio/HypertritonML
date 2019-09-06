#!/usr/bin/env python3
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="Do the training", action="store_true")
parser.add_argument("-o", "--optimize", help="Run the optimization", action="store_true")
parser.add_argument("-s", "--significance", help="Run the significance optimisation studies", action="store_true")
parser.add_argument("config", help="Path to the YAML configuration file")
args = parser.parse_args()

import collections.abc
import os
import time
import warnings

import analysis_utils as au
import generalized_analysis as ga
import pandas as pd
import xgboost as xgb
from generalized_analysis import GeneralizedAnalysis

import yaml
import json


# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

HYPERPARAMS = {'eta': 0.05,
               'min_child_weight': 8,
               'max_depth': 10,
               'gamma': 0.7,
               'subsample': 0.8,
               'colsample_bytree': 0.9,
               'scale_pos_weight': 10}

signal_selection = '2<=HypCandPt<=10'
backgound_selection = '(InvMass<2.98 or InvMass>3.005) and HypCandPt<=10'

mc_path = os.path.expandvars(params['MC_PATH'])
data_path = os.path.expandvars(params['DATA_PATH'])

analysis = GeneralizedAnalysis(params['NBODY'], mc_path, data_path,
                               signal_selection, backgound_selection, 
                               cent_class=params['CENTRALITY_CLASS'])

# start timer for performance evaluation
start_time = time.time()

bdt_efficiency = []
score_selection = []

if os.path.isfile('mydirectory/myfile.txt') and not args.significance: 
    with open(os.environ['HYPERML_DATA_2']+'/significance_data.txt') as json_file:
        data = json.load(json_file)
    score_selection = data['eff_bdt']

if not os.path.isfile('mydirectory/myfile.txt') and not args.significance:
    args.significance = True # does it work?

significance_results = {}

for cclass in params['CENTRALITY_CLASS']:
    for ptbin in params['PT_BINS']:
        for ctbin in params['CT_BINS']:
            print('============================================')
            print('centrality class: ', cclass)
            print('ct bin: ', ctbin)
            print('pt bin: ', ptbin)

            part_time = time.time()

            # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
            data = analysis.prepare_dataframe(
                params['TRAINING_COLUMNS'],
                cclass, ct_range=ctbin, pt_range=ptbin, test=True)

            if args.train:
                print("Training model...")
                # train and test the model with some performance plots
                model = analysis.train_test_model(
                    data, params['TRAINING_COLUMNS'], params['XGBOOST_PARAMS'],
                    hyperparams=HYPERPARAMS, ct_range=ctbin,
                    cent_class=cclass, pt_range=ptbin, optimize=args.optimize, optimize_mode='gs',
                    num_rounds=500, es_rounds=20)
                print('--- model trained in {:.4f} minutes ---\n'.format((time.time() - part_time) / 60))
                analysis.save_model(model, ct_range=ctbin, cent_class=cclass, pt_range=ptbin)
                print('Model saved\n')
            else:
                model = analysis.load_model(ct_range=ctbin, cent_class=cclass, pt_range=ptbin)
            
            if args.significance:
                score,bdt_eff = analysis.significance_scan(data[2:3],model,params['TRAINING_COLUMNS'], ct_range=ctbin,
                    pt_range=ptbin, cent_class=cclass,
                    custom=params['MAX_SIGXEFF'], n_points=100)
                score_selection.append(score)
            
        
            dtest = xgb.DMatrix(data=(data[2][params['TRAINING_COLUMNS']]))
            y_pred = model.predict(dtest, output_margin=True)

            data[2].eval('Score = @y_pred', inplace=True)
            data[2].eval('y = @data[3]', inplace=True)
            bdt_efficiency.append(analysis.bdt_efficiency(data[2],score_selection[cclass*len(params['CENTRALITY_CLASS'])+ptbin*len(params['PT_BINS'])+ctbin]))

            #the real analysis is still missing


# if args.significance:
#     save['cut_score'] = score_selection
#     with open(os.environ['HYPERML_DATA_2']+'/significance_data.txt', 'w') as json_file:
#         json.dump(save, json_file)            
            
                
                
# print execution time to performance evaluation
print('')
print('--- {:.4f} minutes ---'.format((time.time() - start_time) / 60))