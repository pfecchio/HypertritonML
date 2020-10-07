#!/usr/bin/env python3
#macro to compute the shift of the gaussian mu parameter due to reconstruction and BDT selection
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
from ROOT import TFile, gROOT, TF1, TH1D, TH2D, TCanvas, TLegend

from array import*

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
LARGE_DATA = params['LARGE_DATA']
LOAD_LARGE_DATA = params['LOAD_LARGE_DATA']

CENT_CLASSES = params['CENTRALITY_CLASS']
CT_BINS = params['CT_BINS']
PT_BINS = params['PT_BINS']

COLUMNS = params['TRAINING_COLUMNS']
MODEL_PARAMS = params['XGBOOST_PARAMS']
HYPERPARAMS = params['HYPERPARAMS']
HYPERPARAMS_RANGE = params['HYPERPARAMS_RANGE']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)

SPLIT_MODE = args.split

if SPLIT_MODE:
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

resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]
file_name =  resultsSysDir + '/' + params['FILE_PREFIX'] + '_mass_shift.root'
results_file = TFile(file_name,"recreate")
file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_results_fit.root'
eff_file = TFile(file_name, 'read')

results_file.cd()
#efficiency from the significance scan

SEL_EFF = []
gauss = TF1('gauss','gaus')

for split in SPLIT_LIST:
    BDTeff2D = eff_file.Get('0-90'+split+'/BDTeff')
    
    if 'ct' in params['FILE_PREFIX']:
        BINS = params['CT_BINS']
        BDTeff = BDTeff2D.ProjectionY("BDTeff", 1, BDTeff2D.GetNbinsX())
        for iBin in range(1,BDTeff.GetNbinsX()+1):
            SEL_EFF.append(BDTeff.GetBinContent(iBin))
        xlabel = '#it{c}t (cm)'
    else:
        BINS = params['PT_BINS']
        BDTeff = BDTeff2D.ProjectionX("BDTeff", 1, BDTeff2D.GetNbinsY())
        for iBin in range(1,BDTeff.GetNbinsX()+1):
            SEL_EFF.append(BDTeff.GetBinContent(iBin))
        xlabel = '#it{p}_{T} (GeV/#it{c})'
    if split == '_matter':
        title = 'matter'
    else:
        title = 'antimatter'
    binning = array('d',BINS)
    canvas = TCanvas('canvas'+split,"")
    mean_shift = TH2D('mean_shift'+split, title+';'+xlabel+';BDT efficiency; mean[m_{after BDT}]-m_{gen}] (MeV/c^{2})',len(BINS)-1,binning,68,0.195,0.995)
    opt_shift = TH1D('opt_shift'+split, title+';'+xlabel+'; mean[m_{after BDT}]-m_{gen} (MeV/c^{2})',len(BINS)-1,binning)
    sigmaMC = TH1D('sigmaMC'+split, title+';'+xlabel+'Monte Carlo; #sigma (MeV/c^{2})',len(BINS)-1,binning)
    fit_shift = TH2D('fit_shift'+split, title+';'+xlabel+';BDT efficiency; #mu_{after BDT}-m_{gen} (MeV/c^{2})',len(BINS)-1,binning,68,0.195,0.995)
    opt_fit_shift = TH1D('opt_fit_shift'+split, title+';'+xlabel+'; #mu_{after BDT}-m_{gen} (MeV/c^{2})',len(BINS)-1,binning)

    ml_analysis = TrainingAnalysis(N_BODY, signal_path, bkg_path, split)
    
                    
    application_columns = ['score', 'm', 'ct', 'pt', 'centrality','ArmenterosAlpha']
    if LARGE_DATA:
        if LOAD_LARGE_DATA:
            df_skimmed = pd.read_parquet(os.path.dirname(data_path) + '/skimmed_df.parquet.gzip')
        else:
            df_skimmed = hau.get_skimmed_large_data(data_path, CENT_CLASSES, PT_BINS, CT_BINS, COLUMNS, application_columns, N_BODY)#, split)
            df_skimmed.to_parquet(os.path.dirname(data_path) + '/skimmed_df.parquet.gzip', compression='gzip')
        
        ml_application = ModelApplication(N_BODY, data_path, analysis_res_path, CENT_CLASSES, split, df_skimmed)
    else:
        ml_application = ModelApplication(N_BODY, data_path, analysis_res_path, CENT_CLASSES, split)
    

    shift_bin = 1
    eff_index=0

    for cclass in CENT_CLASSES:
        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
            for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):

                # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
                data = ml_analysis.prepare_dataframe(COLUMNS, cent_class=cclass, ct_range=ctbin, pt_range=ptbin)
                data2 = data[2]
                data3 = data[3]
                del data

                input_model = xgb.XGBClassifier()
                model_handler = ModelHandler(input_model)
                
                info_string = f'_{cclass[0]}{cclass[1]}_{ptbin[0]}{ptbin[1]}_{ctbin[0]}{ctbin[1]}{split}'
                filename_handler = handlers_path + '/model_handler' + info_string + '.pkl'
                model_handler.load_model_handler(filename_handler)
                y_pred = model_handler.predict(data2)
                data2 = pd.concat([data2, data3], axis=1, sort=False)
                data2.insert(0, 'score', y_pred)
                del data3

                mass_bins = 40 if ctbin[1] < 16 else 36
        
                eff_score_array, model_handler = ml_application.load_ML_analysis(cclass, ptbin, ctbin, split)
                
                iEff = 1
                for eff, tsd in zip(pd.unique(eff_score_array[0][::-1]), pd.unique(eff_score_array[1][::-1])):
                    #after selection
                    mass_array = np.array(data2.query('y>0.5')['m'].values, dtype=np.float64)
                    counts, roba = np.histogram(mass_array, bins=mass_bins, range=[2.96, 3.05])
                    
                    histo_name = 'selected_' + info_string
                    h1_sel = hau.h1_invmass(counts, cclass, ptbin, ctbin, bins=mass_bins, name=histo_name)
                    h1_sel.Draw()
                    h1_sel.Fit(gauss,"Q")
                    
                    
                    mu_sel = gauss.GetParameter(1)
                    err_mu_sel = gauss.GetParError(1)
                    
                    if round(eff,2)==0.80:
                        h1_sel.Write()

                    mean_shift.SetBinContent(shift_bin,iEff,(h1_sel.GetMean()-hyp3mass)*1000)
                    mean_shift.SetBinError(shift_bin,iEff,h1_sel.GetMeanError()*1000)
                    

                    sigmaMC.SetBinContent(shift_bin,iEff,gauss.GetParameter(2)*1000)
                    sigmaMC.SetBinError(shift_bin,iEff,gauss.GetParError(2)*1000)
                    
                    fit_shift.SetBinContent(shift_bin,iEff,(mu_sel-hyp3mass)*1000)
                    fit_shift.SetBinError(shift_bin,iEff,err_mu_sel*1000)
                    
                    #round because the eff are in the format x.xxxxxxxx04
                    if round(eff,2)==round(SEL_EFF[eff_index],2):
                        print(mu_sel,"+-",err_mu_sel)
                        opt_shift.SetBinContent(shift_bin,(h1_sel.GetMean()-hyp3mass)*1000.)
                        opt_shift.SetBinError(shift_bin,h1_sel.GetMeanError()*1000.)
                      
                        opt_fit_shift.SetBinContent(shift_bin,(mu_sel-hyp3mass)*1000.)
                        opt_fit_shift.SetBinError(shift_bin,err_mu_sel*1000.)
                        
                    iEff = iEff+1

                eff_index = eff_index+1
                shift_bin = shift_bin+1
        

        del ml_analysis
        del ml_application
    mean_shift.Write()
    sigmaMC.SetMarkerStyle(20)
    sigmaMC.Write()
    opt_shift.Write()
    fit_shift.Write()
    opt_fit_shift.Write()
    opt_shift.SetMarkerStyle(20)
    opt_shift.SetMarkerColor(2)
    opt_shift.SetLineColor(2)
    opt_fit_shift.SetMarkerStyle(20)
    opt_fit_shift.SetMarkerColor(4)
    opt_fit_shift.SetLineColor(4)
    opt_shift.SetTitle(';'+xlabel+';#delta_{MC} (MeV/c^{2})')
    opt_shift.Draw()
    opt_fit_shift.Draw("same")
    legend = TLegend(0.7, 0.7,1, 1)
  
    legend.AddEntry(opt_shift,'#Deltamean[m]', "lep")
    legend.AddEntry(opt_fit_shift,'#Delta#mu from gaussian fit', "lep")
    legend.Draw()
    canvas.Write()
