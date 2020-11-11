#!/usr/bin/env python3

import argparse
import math
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
from hipe4ml import analysis_utils, plot_utils
from hipe4ml.model_handler import ModelHandler

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
ROOT.gROOT.SetBatch()

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

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)

SPLIT_LIST = ['_matter','_antimatter'] if args.split else ['']

HYPERTRITON_MASS = 2.9913100
MASS_BINS = 18

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
bkg_path = os.path.expandvars(params['BKG_PATH'])
data_path = os.path.expandvars(params['DATA_PATH'])
analysis_res_path = os.path.expandvars(params['ANALYSIS_RESULTS_PATH'])

handlers_path = os.environ['HYPERML_MODELS_{}'.format(N_BODY)] + '/handlers'
###############################################################################

###############################################################################
# input/output files
results_dir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

file_name =  results_dir + '/' + params['FILE_PREFIX'] + '_mass_shift.root'
results_file = ROOT.TFile(file_name, 'recreate')

file_name = results_dir + f'/Efficiencies/{FILE_PREFIX}_sigscan.npy'
sigscan_dict = np.load(file_name, allow_pickle=True).item()

results_file.cd()
###############################################################################

###############################################################################
# define RooFit objects
mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2.975, 3.01, 'GeV/c^{2}')
hyp_mass_mc = ROOT.RooRealVar('hyp_mass_mc', 'hyp_mass_mc', 2.989, 2.993, 'GeV^-1')
width_hyp_mc = ROOT.RooRealVar('width', 'hyp_mass_mc width',0.0005, 0.004, 'GeV/c^2')
n1 = ROOT.RooRealVar('n1', 'n1 const', 0., 1., 'GeV')
width_res = ROOT.RooRealVar('tails width', 'tails width', 0.002, 0.006, 'GeV')
hyp_pdf = ROOT.RooGaussian('hyp_mc_sigma', 'hyp_mc_sigma', mass, hyp_mass_mc, width_hyp_mc)
res_sig_pdf = ROOT.RooGaussian('res_sig', 'signal + resolution', mass, hyp_mass_mc, width_res)
conv_sig = ROOT.RooAddPdf('conv_sig', 'Double Gaussian', ROOT.RooArgList(hyp_pdf, res_sig_pdf), ROOT.RooArgList(n1))
###############################################################################

xlabel = '#it{p}_{T} (GeV/#it{c})'

for split in SPLIT_LIST:
    title = '(anti-)matter'
    if split is not '':
        title = split.replace('_', '')

    pt_binning = np.array(BINS, 'double')
    n_pt_bins = len(BINS) - 1
    
    # shift due to reconstruction only
    shift_fit_nobdt = ROOT.TH1D('shift_fit_nobdt' + split, title + ';' + xlabel + '; m_{rec}-m_{gen} (MeV/c^{2})',
                                 n_pt_bins, pt_binning)
    
    # shift due to reco+BDT with mean MC mass
    shift_mean = ROOT.TH2D('shift_mean' + split, title + ';' + xlabel + ';BDT efficiency; mean[m_{after BDT}-m_{gen}] (MeV/c^{2})',
                            n_pt_bins, pt_binning, len(EFF_ARRAY), EFF_MIN-0.005, EFF_MAX-0.005)

    # shift due to reco+BDT with MC mass fit
    shift_fit = ROOT.TH2D('shift_fit' + split, title + ';' + xlabel + ';BDT efficiency; #mu_{after BDT}-m_{gen} (MeV/c^{2})',
                           n_pt_bins, pt_binning, len(EFF_ARRAY), EFF_MIN-0.005, EFF_MAX-0.005)

    # MC sigma after reco+BDT
    sigma_mc = ROOT.TH2D('sigma_mc' + split, title + ';' + xlabel + ';BDT efficiency; #sigma_{after BDT} (MeV/c^{2})',
                          n_pt_bins, pt_binning, len(EFF_ARRAY), EFF_MIN-0.005, EFF_MAX-0.005)
    
    ml_analysis = TrainingAnalysis(N_BODY, signal_path, bkg_path, split)
    ml_application = ModelApplication(N_BODY, data_path, analysis_res_path, CENT_CLASSES, split)
    
    shift_bin = 1

    for cclass in CENT_CLASSES:
        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
            for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                # use the whole MC for the shift estimation
                data = ml_analysis.prepare_dataframe(COLUMNS, cent_class=cclass, ct_range=ctbin, pt_range=ptbin, test_size=0.9999)

                input_model = xgb.XGBClassifier()
                model_handler = ModelHandler(input_model)
                
                info_string = f'_{cclass[0]}{cclass[1]}_{ptbin[0]}{ptbin[1]}_{ctbin[0]}{ctbin[1]}{split}'
                filename_handler = handlers_path + '/model_handler' + info_string + '.pkl'
                model_handler.load_model_handler(filename_handler)
                
                y_pred = model_handler.predict(data[2])
                test_set = pd.concat([data[2], data[3]], axis=1, sort=False)
                test_set.insert(0, 'score', y_pred)
                test_set.query('y>0', inplace=True)

                # fit the reconstructed MC hypertritons before BDT selection
                mass_array = np.array(test_set['m'].values, dtype=np.float64)
                roo_mass = hau.ndarray2roo(mass_array, mass)

                # actual fit with signal+resolution function
                conv_sig.fitTo(roo_mass)

                mu_sel = hyp_mass_mc.getVal()
                mu_sel_error = hyp_mass_mc.getError()

                shift_fit_nobdt.SetBinContent(shift_bin, (mu_sel-HYPERTRITON_MASS)*1000)
                shift_fit_nobdt.SetBinError(shift_bin, mu_sel_error * 1000)
                
                # plot for the mc mass fit before the BDT selections
                xframe = mass.frame(ROOT.RooFit.Name('mass_reco_hyp'), ROOT.RooFit.Bins(MASS_BINS))
                roo_mass.plotOn(xframe)
                conv_sig.plotOn(xframe)
                conv_sig.paramOn(xframe)
                xframe.Write()
        
                eff_score_array, model_handler = ml_application.load_ML_analysis(cclass, ptbin, ctbin, split)
                
                # loop on the BDT efficiency
                eff_index = 1
                for eff, tsd in zip(pd.unique(eff_score_array[0][::-1]), pd.unique(eff_score_array[1][::-1])):
                    # Fit the BDT selected MC hypertritons at fixed efficiency        
                    mass_array = np.array(test_set.query('score>@tsd')['m'].values, dtype=np.float64)
                    roo_mass = hau.ndarray2roo(mass_array, mass)

                    # actual fit with signal+resolution function
                    conv_sig.fitTo(roo_mass)
                    
                    mu_sel = hyp_mass_mc.getVal()
                    mu_sel_error = hyp_mass_mc.getError()

                    width_mc = width_hyp_mc.getVal()
                    width_mc_error = width_hyp_mc.getError()

                    shift_mean.SetBinContent(shift_bin, eff_index, (roo_mass.mean(mass)-HYPERTRITON_MASS)*1000)
                    shift_mean.SetBinError(shift_bin, eff_index, 0)

                    sigma_mc.SetBinContent(shift_bin, eff_index, width_mc*1000)
                    sigma_mc.SetBinError(shift_bin, eff_index, width_mc_error*1000)

                    shift_fit.SetBinContent(shift_bin, eff_index, (mu_sel-HYPERTRITON_MASS)*1000)
                    shift_fit.SetBinError(shift_bin, eff_index, mu_sel_error * 1000)

                    eff_index += 1
                        
                shift_bin += 1

    shift_fit_nobdt.Write()
    shift_mean.Write()
    shift_fit.Write()
    sigma_mc.Write()
