#!/usr/bin/env python3
import argparse
import os
import time
import warnings
import math
import numpy as np
import yaml

import hyp_analysis_utils as hau
import pandas as pd
import xgboost as xgb
from analysis_classes import (ModelApplication, TrainingAnalysis)
from hipe4ml import analysis_utils, plot_utils
from hipe4ml.model_handler import ModelHandler
from ROOT import TFile, gROOT

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()



###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='Do the training', action='store_true')
parser.add_argument('--test', help='Training with reduced number of candidates for testing functionalities', action='store_true')
parser.add_argument('-o', '--optimize', help='Run the optimization', action='store_true')
parser.add_argument('-a', '--application', help='Apply ML predictions on data', action='store_true')
parser.add_argument('-s', '--significance', help='Run the significance optimisation studies', action='store_true')
parser.add_argument('-side', '--side', help='Use the sideband as background', action='store_true')
parser.add_argument('-u', '--unbinned', help='Unbinned fit', action='store_true')
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
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
LOAD_LARGE_DATA = params['LOAD_LARGE_DATA']

CENT_CLASSES = params['CENTRALITY_CLASS']
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']

COLUMNS = params['TRAINING_COLUMNS']
MODEL_PARAMS = params['XGBOOST_PARAMS']
HYPERPARAMS = params['HYPERPARAMS']
HYPERPARAMS_RANGE = params['HYPERPARAMS_RANGE']


BKG_MODELS = params['BKG_MODELS']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)

SIGMA_MC = params['SIGMA_MC']

TRAIN = args.train
TEST_MODE = args.test
SPLIT_MODE = args.split
OPTIMIZE = args.optimize
APPLICATION = args.application
SIGNIFICANCE_SCAN = args.significance

if SPLIT_MODE:
    SPLIT_LIST = ['_matter','_antimatter']
else:
    SPLIT_LIST = ['']

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
        ml_analysis = TrainingAnalysis(N_BODY, signal_path, bkg_path, split, sidebands=args.side)
        print(f'--- analysis initialized in {((time.time() - start_time) / 60):.2f} minutes ---\n')

        for cclass in CENT_CLASSES:
            ml_analysis.preselection_efficiency(cclass, CT_BINS, PT_BINS, split)

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
                        model_handler.optimize_params_bayes(
                            data, HYPERPARAMS_RANGE, 'roc_auc', init_points=10, n_iter=10)

                    model_handler.train_test_model(data)
                    print("train test model")
                    print(f'--- model trained and tested in {((time.time() - part_time) / 60):.2f} minutes ---\n')

                    y_pred = model_handler.predict(data[2])
                    data[2].insert(0, 'score', y_pred)
                    eff, tsd = analysis_utils.bdt_efficiency_array(data[3], y_pred, n_points=1000)
                    score_from_eff_array = analysis_utils.score_from_efficiency_array(data[3], y_pred, FIX_EFF_ARRAY)
                    fixed_eff_array = np.vstack((FIX_EFF_ARRAY, score_from_eff_array))

                    if SIGMA_MC:
                        ml_analysis.MC_sigma_array(data, fixed_eff_array, cclass, ptbin, ctbin, split)

                    ml_analysis.save_ML_analysis(model_handler, fixed_eff_array, cent_class=cclass,pt_range=ptbin, ct_range=ctbin, split=split)
                    ml_analysis.save_ML_plots(model_handler, data, [eff, tsd],cent_class=cclass, pt_range=ptbin, ct_range=ctbin, split=split)

        del ml_analysis

    print('')
    print(f'--- training and testing in {((time.time() - start_time) / 60):.2f} minutes ---')

if APPLICATION:
    app_time = time.time()

    file_name = results_dir + f'/{FILE_PREFIX}_results.root'
    results_histos_file = TFile(file_name, 'recreate')

    if(N_BODY==3):
        application_columns = ['score', 'm', 'ct', 'pt', 'centrality', 'positive', 'mppi_vert', 'mppi', 'mdpi', 'tpc_ncls_de', 'tpc_ncls_pr', 'tpc_ncls_pi']
    else:
        application_columns = ['score', 'm', 'ct', 'pt', 'centrality','ArmenterosAlpha']

    if SIGNIFICANCE_SCAN:
        sigscan_results = {}    
    
    if args.unbinned:
        file_name = results_dir + f'/{FILE_PREFIX}_results_unbinned.root'
        results_unbin_file = TFile(file_name, 'recreate')

    for split in SPLIT_LIST:
        if LARGE_DATA:
            if LOAD_LARGE_DATA:
                df_skimmed = pd.read_parquet(os.path.dirname(data_path) + '/skimmed_df.parquet.gzip')
            else:
                df_skimmed = hau.get_skimmed_large_data(data_path, CENT_CLASSES, PT_BINS, CT_BINS, COLUMNS, application_columns, N_BODY, split)
                df_skimmed.to_parquet(os.path.dirname(data_path) + '/skimmed_df.parquet.gzip', compression='gzip')

            ml_application = ModelApplication(N_BODY, data_path, analysis_res_path, CENT_CLASSES, split, df_skimmed)

        else:
            ml_application = ModelApplication(N_BODY, data_path, analysis_res_path, CENT_CLASSES, split)

        for cclass in CENT_CLASSES:
            # create output structure
            cent_dir_histos = results_histos_file.mkdir(f'{cclass[0]}-{cclass[1]}{split}')

            th2_efficiency = ml_application.load_preselection_efficiency(cclass, split)

            df_sign = pd.DataFrame()

            for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
                ptbin_index = ml_application.presel_histo.GetXaxis().FindBin(0.5 * (ptbin[0] + ptbin[1]))

                for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                    ctbin_index = ml_application.presel_histo.GetYaxis().FindBin(0.5 * (ctbin[0] + ctbin[1]))

                    print('\n==================================================')
                    print('centrality:', cclass, ' ct:', ctbin, ' pT:', ptbin, split)
                    print('Application and signal extraction ...', end='\r')
                    mass_bins = 40 if ctbin[1] < 16 else 36

                    presel_eff = ml_application.get_preselection_efficiency(ptbin_index, ctbin_index)
                    eff_score_array, model_handler = ml_application.load_ML_analysis(cclass, ptbin, ctbin, split)

                    if LARGE_DATA:
                        df_applied = ml_application.get_data_slice(cclass, ptbin, ctbin, application_columns)
                    else: 
                        df_applied = ml_application.apply_BDT_to_data(model_handler, cclass, ptbin, ctbin, model_handler.get_training_columns(), application_columns)

                    if SIGNIFICANCE_SCAN:
                        sigscan_eff, sigscan_tsd = ml_application.significance_scan(df_applied, presel_eff, eff_score_array, cclass, ptbin, ctbin, split, mass_bins)
                        eff_score_array = np.append(eff_score_array, [[sigscan_eff], [sigscan_tsd]], axis=1)

                        sigscan_results[f'ct{ctbin[0]}{ctbin[1]}pt{ptbin[0]}{ptbin[1]}{split}'] = sigscan_eff


                    # define subdir for saving invariant mass histograms
                    sub_dir_histos = cent_dir_histos.mkdir(f'ct_{ctbin[0]}{ctbin[1]}') if 'ct' in FILE_PREFIX else cent_dir_histos.mkdir(f'pt_{ptbin[0]}{ptbin[1]}')

                    for eff, tsd in zip(pd.unique(eff_score_array[0][::-1]), pd.unique(eff_score_array[1][::-1])):
                        sub_dir_histos.cd()

                        if eff == sigscan_eff:
                            df_sign = df_sign.append(df_applied.query('score>@tsd'), ignore_index=True, sort=False)

                        
                        mass_array = np.array(df_applied.query('score>@tsd')['m'].values, dtype=np.float64)

                        counts = np.histogram(mass_array, bins=mass_bins, range=[2.96, 3.05])

                        histo_name = f'eff{eff:.2f}'
                        
                        h1_minv = hau.h1_invmass(counts, cclass, ptbin, ctbin, name=histo_name)
                        h1_minv.Write()

                        if eff == sigscan_eff:
                            for bkg_model in BKG_MODELS:
                                hau.unbinned_mass_fit(mass_array, eff, bkg_model, results_unbin_file, cclass, ctbin, ptbin, split)

                        # if args.unbinned:
                        #     for bkg_model in BKG_MODELS:
                        #         hau.unbinned_mass_fit(mass_array, eff, bkg_model, results_unbin_file, cclass, ctbin, ptbin, split)
                                
                    print('Application and signal extraction: Done!\n')

            cent_dir_histos.cd()
            th2_efficiency.Write()

    try:
        sigscan_results = np.asarray(sigscan_results)
        filename_sigscan = results_dir + f'/Efficiencies/{FILE_PREFIX}_sigscan.npy'
        np.save(filename_sigscan, sigscan_results)

    except:
        print('No sigscan, no sigscan results!')

    df_sign.to_parquet(os.path.dirname(data_path) + '/selected_df.parquet.gzip', compression='gzip')
    print (f'--- ML application time: {((time.time() - app_time) / 60):.2f} minutes ---')
    
    results_histos_file.Close()

    if args.unbinned:
        results_unbin_file.Close()

    print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')
