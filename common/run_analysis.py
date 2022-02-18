#!/usr/bin/env python3
import argparse
import os
import time
import warnings

import hyp_analysis_utils as hau
import numpy as np
import pandas as pd
import ROOT
import xgboost as xgb
import yaml
from analysis_classes import ModelApplication, TrainingAnalysis
from hipe4ml import analysis_utils
from hipe4ml.model_handler import ModelHandler

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
ROOT.gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='Do the training', action='store_true')
parser.add_argument('-o', '--optimize', help='Run the optimization', action='store_true')
parser.add_argument('-a', '--application', help='Apply ML predictions on data', action='store_true')
parser.add_argument('-s', '--significance', help='Run the significance optimisation studies', action='store_true')
parser.add_argument('-side', '--side', help='Use the sideband as background', action='store_true')
parser.add_argument('-matter', '--matter', help='Run with matter', action='store_true')
parser.add_argument('-antimatter', '--antimatter', help='Run with antimatter', action='store_true')


parser.add_argument('config', help='Path to the YAML configuration file')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
###############################################################################

###############################################################################
# define analysis global variables
N_BODY = params['NBODY']
FILE_PREFIX = params['FILE_PREFIX']
LARGE_DATA = params['LARGE_DATA']
LOAD_APPLIED_DATA = params['LOAD_APPLIED_DATA']

CENT_CLASSES = params['CENTRALITY_CLASS']
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']

COLUMNS = params['TRAINING_COLUMNS']
MODEL_PARAMS = params['XGBOOST_PARAMS']
HYPERPARAMS = params['HYPERPARAMS']
HYPERPARAMS_RANGE = params['HYPERPARAMS_RANGE']
BKG_MODELS = params['BKG_MODELS']
MAG_FIELD = params['MAG_FIELD']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
EFF_ARRAY = np.around(np.arange(EFF_MIN, EFF_MAX+EFF_STEP, EFF_STEP), 2)

TRAIN = args.train
OPTIMIZE = args.optimize
APPLICATION = args.application
SIGNIFICANCE_SCAN = args.significance
SIDEBANDS = args.side

SPLIT_LIST = ['']

if args.matter:
    SPLIT_LIST = ['_matter']

if args.antimatter:
    SPLIT_LIST = ['_antimatter']

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
bkg_path = os.path.expandvars(params['BKG_PATH'])
data_path = os.path.expandvars(params['DATA_PATH'])
analysis_res_path = os.path.expandvars(params['ANALYSIS_RESULTS_PATH'])
results_dir = os.environ[f'HYPERML_RESULTS_{N_BODY}']

###############################################################################
start_time = time.time()                          # for performances evaluation

if TRAIN:
    for split in SPLIT_LIST:
        ml_analysis = TrainingAnalysis(N_BODY, signal_path, bkg_path, split, sidebands=SIDEBANDS)
        print(f'--- analysis initialized in {((time.time() - start_time) / 60):.2f} minutes ---\n')

        for cclass in CENT_CLASSES:
            ml_analysis.preselection_efficiency(cclass, CT_BINS, PT_BINS, split, prefix=FILE_PREFIX)

            for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
                for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                    print('\n==================================================')
                    print('centrality:', cclass, ' ct:', ctbin, ' pT:', ptbin, split)

                    part_time = time.time()

                    # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
                    data = ml_analysis.prepare_dataframe(COLUMNS, cent_class=cclass, ct_range=ctbin, pt_range=ptbin)
                    input_model = xgb.XGBClassifier()
                    model_handler = ModelHandler(input_model)
                    
                    model_handler.set_model_params(MODEL_PARAMS)
                    model_handler.set_model_params(HYPERPARAMS)
                    model_handler.set_training_columns(COLUMNS)

                    if OPTIMIZE:
                        model_handler.optimize_params_bayes(data, HYPERPARAMS_RANGE, 'roc_auc', init_points=10, n_iter=10)

                    model_handler.train_test_model(data)

                    print("train test model")
                    print(f'--- model trained and tested in {((time.time() - part_time) / 60):.2f} minutes ---\n')

                    y_pred = model_handler.predict(data[2])
                    data[2].insert(0, 'score', y_pred)

                    if split != "":
                        mc_truth = data[3][data[2]['Matter'] > 0.5] if split=="_matter" else data[3][data[2]['Matter'] < 0.5]
                        y_pred = y_pred[data[2]['Matter'] > 0.5] if split=="_matter" else y_pred[data[2]['Matter'] < 0.5]
                    
                    else:
                        mc_truth = data[3]


                    eff, tsd = analysis_utils.bdt_efficiency_array(mc_truth, y_pred, n_points=1000)
                    score_from_eff_array = analysis_utils.score_from_efficiency_array(mc_truth, y_pred, EFF_ARRAY)
                    fixed_eff_array = np.vstack((EFF_ARRAY, score_from_eff_array))

                    ml_analysis.save_ML_analysis(model_handler, fixed_eff_array, cent_class=cclass, pt_range=ptbin, ct_range=ctbin, split=split)
                    ml_analysis.save_ML_plots(model_handler, data, [eff, tsd], cent_class=cclass, pt_range=ptbin, ct_range=ctbin, split=split)

        del ml_analysis

    print('')
    print(f'--- training and testing in {((time.time() - start_time) / 60):.2f} minutes ---')

if APPLICATION:
    app_time = time.time()

    sigscan_results = {}    

    if (N_BODY==3):
        application_columns = ['score', 'm', 'ct', 'pt', 'centrality', 'positive', 'mppi_vert', 'mppi', 'mdpi', 'tpc_ncls_de', 'tpc_ncls_pr', 'tpc_ncls_pi']
    else:
        application_columns = ['score', 'm', 'ct', 'pt', 'centrality', 'Matter', 'magField']

    print('\n==================================================')
    print('Application and signal extraction ...', end='\r')

    for split in SPLIT_LIST:


        if LOAD_APPLIED_DATA:
            df_applied = pd.read_parquet(os.path.dirname(data_path) + f'/applied_df_{FILE_PREFIX}{split}.parquet.gzip', engine='fastparquet')
            df_applied_mc = pd.read_parquet(os.path.dirname(signal_path) + f'/applied_mc_df_{FILE_PREFIX}{split}.parquet.gzip', engine='fastparquet')
        else:
            df_applied = hau.get_skimmed_data(data_path, CENT_CLASSES, PT_BINS, CT_BINS, COLUMNS, application_columns, N_BODY, split, LARGE_DATA)
            df_applied_mc = hau.get_applied_mc(signal_path, CENT_CLASSES, PT_BINS, CT_BINS, COLUMNS, application_columns, N_BODY, split)

            if MAG_FIELD!="":
                df_applied = df_applied.query(f'magField=={MAG_FIELD}')
                df_applied_mc = df_applied_mc.query(f'magField=={MAG_FIELD}')

            if split == '_antimatter':
                df_applied = df_applied.query('Matter < 0.5')
                df_applied_mc = df_applied_mc.query('Matter < 0.5')
            
            if split == '_matter':
                df_applied = df_applied.query('Matter > 0.5')
                df_applied_mc = df_applied_mc.query('Matter > 0.5')

            df_applied.to_parquet(os.path.dirname(data_path) + f'/applied_df_{FILE_PREFIX}{split}.parquet.gzip', compression='gzip')
            df_applied_mc.to_parquet(os.path.dirname(signal_path) + f'/applied_mc_df_{FILE_PREFIX}{split}.parquet.gzip', compression='gzip')
            
        ml_application = ModelApplication(N_BODY, df_applied, analysis_res_path, CENT_CLASSES, split)

        print('Application and signal extraction: Done!\n')
        
        if SIGNIFICANCE_SCAN:
            for cclass in CENT_CLASSES:
                th2_efficiency = ml_application.load_preselection_efficiency(cclass, split, prefix=FILE_PREFIX)

                for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
                    ptbin_index = ml_application.presel_histo.GetXaxis().FindBin(0.5 * (ptbin[0] + ptbin[1]))

                    for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                        ctbin_index = ml_application.presel_histo.GetYaxis().FindBin(0.5 * (ctbin[0] + ctbin[1]))

                        mass_bins = 70

                        presel_eff = ml_application.get_preselection_efficiency(ptbin_index, ctbin_index)
                        eff_score_array, model_handler = ml_application.load_ML_analysis(cclass, ptbin, ctbin, split)

                        data_slice = ml_application.get_data_slice(cclass, ptbin, ctbin, application_columns)

                        sigscan_eff, sigscan_tsd = ml_application.significance_scan(data_slice, presel_eff, eff_score_array, cclass, ptbin, ctbin, split, mass_bins)
                        eff_score_array = np.append(eff_score_array, [[sigscan_eff], [sigscan_tsd]], axis=1)

                        sigscan_results[f'ct{ctbin[0]}{ctbin[1]}pt{ptbin[0]}{ptbin[1]}'] = [sigscan_eff, sigscan_tsd]

            sigscan_results = np.asarray(sigscan_results)
            filename_sigscan = results_dir + f'/Efficiencies/{FILE_PREFIX}{split}_sigscan.npy'
            np.save(filename_sigscan, sigscan_results)

    print (f'--- ML application time: {((time.time() - app_time) / 60):.2f} minutes ---')

    print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')
