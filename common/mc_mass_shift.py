#!/usr/bin/env python3

import argparse
import os
import time
import warnings
import math
import numpy as np
import yaml

import numpy as np
from scipy.stats import gaussian_kde

import hyp_analysis_utils as hau
import pandas as pd
import xgboost as xgb
from analysis_classes import ModelApplication, TrainingAnalysis
from hipe4ml import analysis_utils, plot_utils
from hipe4ml.model_handler import ModelHandler
from ROOT import TFile, gROOT, TF1, TH1D, TH2D, TCanvas, TLegend

hyp3mass = 2.99131

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
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

CENT_CLASSES = params['CENTRALITY_CLASS']
CT_BINS = params['CT_BINS']
PT_BINS = params['PT_BINS']
BINS = params['PT_BINS']

COLUMNS = params['TRAINING_COLUMNS']
MODEL_PARAMS = params['XGBOOST_PARAMS']
HYPERPARAMS = params['HYPERPARAMS']
HYPERPARAMS_RANGE = params['HYPERPARAMS_RANGE']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)

if args.split:
    SPLIT_LIST = ['_antimatter','_matter']
else:
    SPLIT_LIST = ['']

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
bkg_path = os.path.expandvars(params['BKG_PATH'])
data_path = os.path.expandvars(params['DATA_PATH'])
analysis_res_path = os.path.expandvars(params['ANALYSIS_RESULTS_PATH'])

handlers_path = os.environ['HYPERML_MODELS_{}'.format(N_BODY)]+'/handlers'
###############################################################################

results_dir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

file_name =  results_dir + '/' + params['FILE_PREFIX'] + '_mass_shift.root'
results_file = TFile(file_name,"recreate")

file_name = results_dir + f'/Efficiencies/{FILE_PREFIX}_sigscan.npy'
sigscan_dict = np.load(file_name).item()

results_file.cd()

tf1_fit = TF1('gauss','gaus')

xlabel = '#it{p}_{T} (GeV/#it{c})'

for split in SPLIT_LIST:
    if split is '_matter':
        title = 'matter'

    elif split is '_antimatter':
        title = 'antimatter'

    else:
        title = '(anti-)matter'

    pt_binning = np.array(BINS, 'double')
    n_pt_bins = len(BINS)-1
    
    mean_shift = TH2D('mean_shift'+split, title+';'+xlabel+';BDT efficiency; mean[m_{after BDT}]-m_{gen}] (MeV/c^{2})', n_pt_bins, pt_binning, 68, 0.195, 0.995)
    opt_shift = TH1D('opt_shift'+split, title+';'+xlabel+'; mean[m_{after BDT}]-m_{gen} (MeV/c^{2})', n_pt_bins, pt_binning)
    sigma_mc = TH2D('sigma_mc'+split, title+';'+xlabel+';BDT efficiency; #sigma_{after BDT} (MeV/c^{2})', n_pt_bins, pt_binning, 68, 0.195, 0.995)
    opt_sigma_mc = TH1D('opt_sigma_mc'+split, title+';'+xlabel+'Monte Carlo; #sigma (MeV/c^{2})', n_pt_bins, pt_binning)
    fit_shift = TH2D('fit_shift'+split, title+';'+xlabel+';BDT efficiency; #mu_{after BDT}-m_{gen} (MeV/c^{2})', n_pt_bins, pt_binning,68,0.195,0.995)
    opt_fit_shift = TH1D('opt_fit_shift'+split, title+';'+xlabel+'; #mu_{after BDT}-m_{gen} (MeV/c^{2})', n_pt_bins, pt_binning)

    ml_analysis = TrainingAnalysis(N_BODY, signal_path, bkg_path, split)
    application_columns = ['score', 'm', 'ct', 'pt', 'centrality', 'ArmenterosAlpha']

    ml_application = ModelApplication(N_BODY, data_path, analysis_res_path, CENT_CLASSES, split)
    
    shift_bin = 1

    for cclass in CENT_CLASSES:
        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
            for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
                data = ml_analysis.prepare_dataframe(COLUMNS, cent_class=cclass, ct_range=ctbin, pt_range=ptbin)

                input_model = xgb.XGBClassifier()
                model_handler = ModelHandler(input_model)
                
                info_string = f'_{cclass[0]}{cclass[1]}_{ptbin[0]}{ptbin[1]}_{ctbin[0]}{ctbin[1]}{split}'
                filename_handler = handlers_path + '/model_handler' + info_string + '.pkl'
                model_handler.load_model_handler(filename_handler)
                
                y_pred = model_handler.predict(data[2])
                test_set = pd.concat([data[2], data[3]], axis=1, sort=False)
                test_set.insert(0, 'score', y_pred)
                test_set.query('y>0', inplace=True)

                mass_bins = 40 if ctbin[1] < 16 else 36
        
                eff_score_array, model_handler = ml_application.load_ML_analysis(cclass, ptbin, ctbin, split)
                
                eff_index = 1
                for eff, tsd in zip(pd.unique(eff_score_array[0][::-1]), pd.unique(eff_score_array[1][::-1])):
                    mass_array = np.array(test_set.query('score>@tsd')['m'].values, dtype=np.float64)                    
                    counts, _ = np.histogram(mass_array, bins=mass_bins, range=[2.96, 3.05])
                    
                    histo_name = 'selected_' + info_string
                    h1_sel = hau.h1_invmass(counts, cclass, ptbin, ctbin, name=histo_name)
                    h1_sel.Draw()
                    h1_sel.Fit(tf1_fit,'Q')
                    
                    mu_sel = tf1_fit.GetMaximum()
                    err_mu_sel = tf1_fit.GetParError(1)

                    mean_shift.SetBinContent(shift_bin, eff_index, (h1_sel.GetMean()-hyp3mass)*1000)
                    mean_shift.SetBinError(shift_bin, eff_index, h1_sel.GetMeanError()*1000)
                    
                    sigma_mc.SetBinContent(shift_bin, eff_index, tf1_fit.GetParameter(2)*1000)
                    sigma_mc.SetBinError(shift_bin, eff_index, tf1_fit.GetParError(2)*1000)
                    
                    fit_shift.SetBinContent(shift_bin, eff_index, (mu_sel-hyp3mass)*1000)
                    fit_shift.SetBinError(shift_bin, eff_index, err_mu_sel*1000)

                    if round(eff, 2) == round(sigscan_dict[f'ct{ctbin[0]}{ctbin[1]}pt{ptbin[0]}{ptbin[1]}{split}'], 2):
                        opt_shift.SetBinContent(shift_bin, eff_index, (h1_sel.GetMean()-hyp3mass)*1000)
                        opt_shift.SetBinError(shift_bin, eff_index, h1_sel.GetMeanError() * 1000)

                        opt_sigma_mc.SetBinContent(shift_bin, eff_index, tf1_fit.GetParameter(2)*1000)
                        opt_sigma_mc.SetBinError(shift_bin, eff_index, tf1_fit.GetParError(2)*1000)

                        opt_fit_shift.SetBinContent(shift_bin, eff_index, (mu_sel-hyp3mass)*1000)
                        opt_fit_shift.SetBinError(shift_bin, eff_index, err_mu_sel*1000)

                        opt_fit_shift.SetBinContent(shift_bin, eff_index, (maximum-hyp3mass)*1000)
                        opt_fit_shift.SetBinError(shift_bin, eff_index, err_mu_sel*1000)

                        h1_sel.Write()

                    eff_index += 1
                        
                shift_bin += 1

        del ml_analysis
        del ml_application

    mean_shift.Write()
    sigma_mc.Write()
    opt_shift.Write()
    fit_shift.Write()
    opt_fit_shift.Write()
    opt_shift.SetMarkerStyle(20)
    opt_shift.SetMarkerColor(2)
    opt_shift.SetLineColor(2)
    opt_fit_shift.SetMarkerStyle(20)
    opt_fit_shift.SetMarkerColor(4)
    opt_fit_shift.SetLineColor(4)
    opt_sigma_mc.SetMarkerStyle(20)
    opt_sigma_mc.SetMarkerColor(4)
    opt_sigma_mc.SetLineColor(4)
    opt_shift.SetTitle(';'+xlabel+';#delta_{MC} (MeV/c^{2})')
    opt_shift.Draw()
    opt_fit_shift.Draw("same")
    legend = TLegend(0.7, 0.7,1, 1)
  
    legend.AddEntry(opt_shift,'#Deltamean[m]', "lep")
    legend.AddEntry(opt_fit_shift,'#Delta#mu from gaussian fit', "lep")
    legend.Draw()
