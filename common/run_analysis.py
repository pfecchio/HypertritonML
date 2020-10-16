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
import ROOT

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
results_dir = os.environ['HYPERML_RESULTS_{}'.format(N_BODY)]

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

                        if(eff==sigscan_eff):
                            df_sign = df_sign.append(df_applied.query('score>@tsd'), ignore_index=True, sort=False)
                        
                        mass_array = np.array(df_applied.query('score>@tsd')['m'].values, dtype=np.float64)
                        counts, _ = np.histogram(mass_array, bins=mass_bins, range=[2.96, 3.05])

                        histo_name = f'eff{eff:.2f}'
                        
                        h1_minv = hau.h1_invmass(counts, cclass, ptbin, ctbin, bins=mass_bins, name=histo_name)
                        h1_minv.Write()

                        #perform an unbinned fit
                        if args.unbinned:
                            for bkgmodel in BKG_MODELS:        
                                print("********************")
                                print(bkgmodel)
                                print("********************")
                              
                                results_unbin_file.cd()
                                mass = ROOT.RooRealVar("m","m_{^{3}He+#pi}",2.97,3.015,"GeV/c^{2}")
                                width = ROOT.RooRealVar("width","B0 mass width",0.001,0.003,"GeV/c^2")
                                mb0 = ROOT.RooRealVar("mb0","B0 mass",2.989,2.993,"GeV^-1")
                                slope = ROOT.RooRealVar("slope","slope mass",-100.,100.,"GeV")
                                sig = ROOT.RooGaussian("sig","B0 sig PDF",mass,mb0,width)
                                c0 = ROOT.RooRealVar("c0","constant c0",1.,"GeV/c^{2}")
                                c1 = ROOT.RooRealVar("c1","constant c1",1.,"GeV/c^{2}")
                                c2 = ROOT.RooRealVar("c2","constant c2",1.,"GeV/c^{2}")

                                n_sig = ROOT.RooRealVar("n1","n1 const",0.,10000,"GeV")  
                                n_bkg = ROOT.RooRealVar("n2","n2 const",0.,10000,"GeV")

                                if bkgmodel == 'pol1':
                                    bkg_func = ROOT.RooPolynomial("bkg","pol1 for bkg",mass,ROOT.RooArgList(mass,c0,c1))                                  
                                elif bkgmodel == 'pol2':
                                    bkg_func = ROOT.RooPolynomial("bkg","pol2 for bkg",mass,ROOT.RooArgList(mass,c0,c1,c2))
                                else:
                                    bkg_func = ROOT.RooExponential("bkg","expo for bkg",mass,slope)

                                sum = ROOT.RooAddPdf("sum","sig+bkg",ROOT.RooArgList(sig,bkg_func),ROOT.RooArgList(n_sig,n_bkg))
                            
                                var = ROOT.RooRealVar("m", "Example Variable",2.97,3.015)
                                roo_data = hau.df2roo(df_applied.query('score>@tsd'), {'m': var})
                                sum.fitTo(roo_data)
                                frame = ROOT.RooPlot()
                                frame = mass.frame(18)
                                roo_data.plotOn(frame)
                                sum.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kRed))
                                sum.plotOn(frame, ROOT.RooFit.Components("bkg"),ROOT.RooFit.LineStyle(9), ROOT.RooFit.LineColor(ROOT.kBlue))
                                #add TPaveText
                                nsigma = 3
                                mu = mb0.getVal()
                                muErr = mb0.getError()
                                sigma = width.getVal()
                                sigmaErr = width.getError()
                                #binwidth hardcoded!!!!
                                xset = ROOT.RooArgSet(mass)
                                int_sig = sig.createIntegral(xset)
                                signal = n_sig.getVal() #/ 0.0025
                                errsignal = math.sqrt(signal)#int_sig.getPropagatedError(sig, xset);
                                mass_int = mass
                                mass_int.setRange("significance range", mu-nsigma*sigma,mu+nsigma*sigma)
                                nset = ROOT.RooArgSet(mass)
                                xset = ROOT.RooArgSet(mass_int)#
                                int_bkg = bkg_func.createIntegral(xset,ROOT.RooFit.NormSet(nset),ROOT.RooFit.Range("significance range"))
                                bkg = int_bkg.getVal()*n_bkg.getVal() #/ 0.0025
                                print("pt: ",ptbin," ct: ",ctbin," eff: ",eff)
                                print("range: [", mu-nsigma*sigma,",",mu+nsigma*sigma,"]")
                                print("sig:",signal," bkg:",bkg)
                                print("sig:",signal*n_sig.getVal()," bkg:",bkg*n_bkg.getVal())
                                print("sig:",n_sig.getVal()," bkg:",n_bkg.getVal())
                                
                                #x = input()
                                if bkg > 0:
                                    errbkg = math.sqrt(bkg)
                                else:
                                    errbkg = 0
                                # compute the significance
                                if signal+bkg > 0:
                                    signif = signal/math.sqrt(signal+bkg)
                                    deriv_sig = 1/math.sqrt(signal+bkg)-signif/(2*(signal+bkg))
                                    deriv_bkg = -signal/(2*(math.pow(signal+bkg, 1.5)))
                                    errsignif = math.sqrt((errsignal*deriv_sig)**2+(errbkg*deriv_bkg)**2)
                                else:
                                    signif = 0
                                    errsignif = 0

                                pinfo = ROOT.TPaveText(0.55, 0.5, 0.95, 0.9, "NDC")
                                pinfo.SetBorderSize(0)
                                pinfo.SetFillStyle(0)
                                pinfo.SetTextAlign(30+3)
                                pinfo.SetTextFont(42)
                                string = f'ALICE Internal, Pb-Pb 2018 {cclass[0]}-{cclass[1]}%'
                                pinfo.AddText(string)
                                
                                decay_label = {
                                    "": ['{}^{3}_{#Lambda}H#rightarrow ^{3}He#pi^{-} + c.c.','{}^{3}_{#Lambda}H#rightarrow dp#pi^{-} + c.c.'],
                                    "_matter": ['{}^{3}_{#Lambda}H#rightarrow ^{3}He#pi^{-}','{}^{3}_{#Lambda}H#rightarrow dp#pi^{-}'],
                                    "_antimatter": ['{}^{3}_{#bar{#Lambda}}#bar{H}#rightarrow ^{3}#bar{He}#pi^{+}','{}^{3}_{#Lambda}H#rightarrow #bar{d}#bar{p}#pi^{+}'],
                                }

                                string = decay_label[split][N_BODY-2]+', %i #leq #it{ct} < %i cm %i #leq #it{p}_{T} < %i GeV/#it{c} ' % (
                                    ctbin[0], ctbin[1], ptbin[0], ptbin[1])
                                pinfo.AddText(string)

                                string = f'#mu {mu*1000:.2f} #pm {muErr*1000:.2f} MeV/c^{2}'
                                pinfo.AddText(string)

                                string = f'#sigma {sigma*1000:.2f} #pm {sigmaErr*1000:.2f} MeV/c^{2}'
                                pinfo.AddText(string)
                                if roo_data.sumEntries()>0:
                                    string = '#chi^{2}/NDF'+f'{frame.chiSquare(6 if bkgmodel=="pol2" else 5):.2f}'
                                    #pinfo.AddText(string)

                                string = f'Significance ({nsigma:.0f}#sigma) {signif:.1f} #pm {errsignif:.1f} '
                                pinfo.AddText(string)

                                string = f'S2 ({nsigma:.0f}#sigma) {signal:.0f} #pm {errsignal:.0f}'
                                pinfo.AddText(string)

                                string = f'B ({nsigma:.0f}#sigma) {bkg:.0f} #pm {errbkg:.0f}'
                                pinfo.AddText(string)

                                if bkg > 0:
                                    ratio = signal/bkg
                                    string = f'S/B ({nsigma:.0f}#sigma) {ratio:.4f}'

                                pinfo.AddText(string)
                                #######
                                frame.addObject(pinfo)
                                frame.Write(f'roo_ct{ctbin[0]}{ctbin[1]}_pT{ptbin[0]}{ptbin[1]}_cen{cclass[0]}{cclass[1]}_eff{eff:.2f}_model{bkgmodel}{split}')
                                width.Write(f'sig_ct{ctbin[0]}{ctbin[1]}_pT{ptbin[0]}{ptbin[1]}_cen{cclass[0]}{cclass[1]}_eff{eff:.2f}_model{bkgmodel}'+split)
                                mb0.Write(f'm_ct{ctbin[0]}{ctbin[1]}_pT{ptbin[0]}{ptbin[1]}_cen{cclass[0]}{cclass[1]}_eff{eff:.2f}_model{bkgmodel}'+split)

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
