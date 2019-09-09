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
import yaml
import json

from array import array
import numpy as np

from ROOT import TFile,TF1,TH1D,TCanvas,TPaveText,gStyle

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

bdt_efficiency = []
score_selection = []
n_hytr = []
preselection_efficiency = []
score_bdteff_array = []

optimisation_params = params['HYPERPARAMS'] if params['OPTIMIZATION_STRATEGY'] == 'gs' else params['HYPERPARAMS_RANGE']
optimisation_strategy = 'gs' if params['OPTIMIZATION_STRATEGY'] == 'gs' else 'bayes'


for cclass in params['CENTRALITY_CLASS']:
    for ptbin in params['PT_BINS']:
        for ctbin in params['CT_BINS']:
            print('============================================')
            print('centrality: ',cclass,' ct: ',ctbin,' pT: ',ptbin)
            part_time = time.time()

            preselection_efficiency.append(analysis.preselection_efficiency(ct_range=ctbin,pt_range=ptbin,cent_class=cclass))

            # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
            data = analysis.prepare_dataframe(
                params['TRAINING_COLUMNS'],
                cclass, ct_range=ctbin, pt_range=ptbin, test=True)

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
                score,bdt_eff = analysis.significance_scan(data[2:4],model,params['TRAINING_COLUMNS'], ct_range=ctbin,
                    pt_range=ptbin, cent_class=cclass,
                    custom=params['MAX_SIGXEFF'], n_points=100)
                score_bdteff_array.append([score, bdt_eff])
                analysis.save_score_eff(score_bdteff_array, ct_range=ctbin, pt_range=ptbin, cent_class=cclass)
            else:
                score_bdteff_array = analysis.load_score_eff(ct_range=ctbin, pt_range=ptbin, cent_class=cclass)
            
            bdt_efficiency = [score_bdteff_array[0][1]]
        
            dtest = xgb.DMatrix(data=(data[2][params['TRAINING_COLUMNS']]))
            y_pred = model.predict(dtest, output_margin=True)

            data[2].eval('Score = @y_pred', inplace=True)
            data[2].eval('y = @data[3]', inplace=True)

            
            if params['SYST_UNCERTANTIES']:
                bdt_efficiency = []
                for shift in params['CUT_SHIFT']:
                    bdt_efficiency.append(analysis.bdt_efficiency(data[2],score_bdteff_array[0][0]+shift))
                    score.append(score_bdteff_array[0][0]+shift)
            else:
                score = [score_bdteff_array[0][0]]
        
            
            if params['BDT_EFF_CUTS']:
                bdt_efficiency = params['BDT_EFFICIENCY']
                score = analysis.score_from_efficiency(model,[data[2],data[3]],bdt_efficiency,params['TRAINING_COLUMNS'],ct_range=ctbin,pt_range=ptbin,cent_class=cclass)
            
            print('score array: ',score)
            print('bdt efficiency array: ',bdt_efficiency)

            for index in range(0,len(score)):
                print('bdt efficiency: ',bdt_efficiency[index])
                #file_name should be improved
                if params['BDT_EFF_CUTS']:
                    file_name = params['FILE_PREFIX'] + '_results_{:.2f}_eff.root'.format(bdt_efficiency[index])
                elif params['SYST_UNCERTANTIES']:
                    file_name = params['FILE_PREFIX'] + '_results_{:.1f}_shift.root'.format(params['CUT_SHIFT'][index])
                else:
                    file_name = params['FILE_PREFIX'] + '_results.root'

                total_cut = '{}<ct<{} and {}<HypCandPt<{} and {}<centrality<{}'.format(
                    ctbin[0], ctbin[1], ptbin[0], ptbin[1], cclass[0], cclass[1])

                dfDataF = analysis.df_data_all.query(total_cut)
                data = xgb.DMatrix(data=(analysis.df_data_all.query(total_cut)[params['TRAINING_COLUMNS']]))
        
                y_pred = model.predict(data,output_margin=True)
                dfDataF.eval('Score = @y_pred',inplace=True)
                Counts,bins = np.histogram(dfDataF.query('Score >@score[@index]')['InvMass'],bins=45,range=[2.96,3.05])
                
                # to save the plots
                recreate=False
                if ctbin==params['CT_BINS'][0]:
                    recreate=True
                    n_hytr.append([])
                # the labels of the plots are still for a ct analysis
                n_hytr[index].append(au.fit(Counts,ctbin,ptbin,cent_class=cclass,recreate=recreate,filename=file_name))
            
            score_selection.append(score)
            #bdt_efficiency.append(bdt_eff)
            #TODO:
            #save n_hytr,score_selection,bdt_efficiency,presel_efficiency
    
# print execution time to performance evaluation
print('')
print('--- {:.4f} minutes ---'.format((time.time() - start_time) / 60))

#TODO:
# -TCanvas has been commented to solve a understood issue
