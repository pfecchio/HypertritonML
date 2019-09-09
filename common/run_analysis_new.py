#!/usr/bin/env python3
import argparse
import collections.abc
import json
import os
import time
import warnings

import yaml

import analysis_utils as au
import generalized_analysis as ga
import pandas as pd
import xgboost as xgb
from generalized_analysis import GeneralizedAnalysis

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="Do the training", action="store_true")
parser.add_argument("-o", "--optimize", help="Run the optimization", action="store_true")
parser.add_argument("-s", "--significance", help="Run the significance optimisation studies", action="store_true")
parser.add_argument("config", help="Path to the YAML configuration file")
args = parser.parse_args()

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

signal_selection = '2<=HypCandPt<=10'
backgound_selection = '(InvMass<2.98 or InvMass>3.005) and HypCandPt<=10'

mc_path = os.path.expandvars(params['MC_PATH'])
data_path = os.path.expandvars(params['DATA_PATH'])

analysis = GeneralizedAnalysis(params['NBODY'], mc_path, data_path,
                               signal_selection, backgound_selection,
                               cent_class=params['CENTRALITY_CLASS'])

# start timer for performance evaluation
start_time = time.time()

score_bdteff_array = [['BDT_eff', 'Score_cut']]

optimisation_params = params['HYPERPARAMS'] if params['OPTIMIZATION_STRATEGY'] == 'gs' else params['HYPERPARAMS_RANGE']
optimisation_strategy = 'gs' if params['OPTIMIZATION_STRATEGY'] == 'gs' else 'bayes'

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
                cclass, ct_range=ctbin, pt_range=ptbin, test=False)

            if args.train:
                # train and test the model with some performance plots
                model = analysis.train_test_model(
                    data, params['TRAINING_COLUMNS'], params['XGBOOST_PARAMS'],
                    hyperparams=optimisation_params, ct_range=ctbin,
                    cent_class=cclass, pt_range=ptbin, optimize=args.optimize,
                    optimize_mode=optimisation_strategy, num_rounds=500, es_rounds=20)
                print('--- model trained in {:.4f} minutes ---\n'.format((time.time() - part_time) / 60))
                analysis.save_model(model, ct_range=ctbin, cent_class=cclass, pt_range=ptbin)
            else:
                model = analysis.load_model(ct_range=ctbin, cent_class=cclass, pt_range=ptbin)

            if args.significance:
                score, bdt_eff = analysis.significance_scan(data[2: 4],
                                                            model, params['TRAINING_COLUMNS'],
                                                            ct_range=ctbin, pt_range=ptbin, cent_class=cclass,
                                                            custom=params['MAX_SIGXEFF'],
                                                            n_points=100)

                score_bdteff_array.append(score, bdt_eff)
                analysis.save_score_eff(score_bdteff_array, ct_range=ctbin, pt_range=ptbin, cent_class=cclass)

            else:
                score_bdteff_array = analysis.load_score_eff(ct_range=ctbin, pt_range=ptbin, cent_class=cclass)


            # the real analysis is still missing


# if args.significance:
#     save['cut_score'] = score_selection
#     with open(os.environ['HYPERML_DATA_2']+'/significance_data.txt', 'w') as json_file:
#         json.dump(save, json_file)


# print execution time to performance evaluation
print('')
print('--- {:.4f} minutes ---'.format((time.time() - start_time) / 60))
