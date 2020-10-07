#!/usr/bin/env python3
import argparse
import os
import time
import warnings
from array import array

import numpy as np
import yaml

import hyp_analysis_utils as hau
import pandas as pd

import ROOT
from ROOT import TFile, gROOT
from analysis_classes import ModelApplication

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('-u', '--unbinned', help='Perform unbinned fit', action='store_true')
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

SPLIT_MODE = args.split
UNBINNED_FIT = args.unbinned

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

BKG_MODELS = params['BKG_MODELS']

results_dir = os.environ['HYPERML_RESULTS_{}'.format(N_BODY)]

###############################################################################
start_time = time.time()                          # for performances evaluation

file_name = results_dir + f'/{FILE_PREFIX}_std_results.root'
results_file = TFile(file_name, 'recreate')

file_name = results_dir + f'/{FILE_PREFIX}_mass_res.root'
shift_file = TFile(file_name, 'read')

standard_selection = 'V0CosPA > 0.9999 and NpidClustersHe3 > 80 and He3ProngPt > 1.8 and pt > 2 and pt < 10 and PiProngPt > 0.15 and He3ProngPvDCA > 0.05 and PiProngPvDCA > 0.2 and TPCnSigmaHe3 < 3.5 and TPCnSigmaHe3 > -3.5 and ProngsDCA < 1'
application_columns = ['NitsClustersHe3','score', 'm', 'ct', 'pt', 'centrality','ArmenterosAlpha','V0CosPA','NpidClustersHe3','He3ProngPt','PiProngPt','He3ProngPvDCA','PiProngPvDCA','TPCnSigmaHe3','ProngsDCA']

for split in SPLIT_LIST:

    if LARGE_DATA:
        if LOAD_LARGE_DATA:
            try:
                df_skimmed = pd.read_parquet(os.path.dirname(data_path) + '/skimmed_df.parquet.gzip')
            except:
                print("you need to run the training")
        else:
            df_skimmed = hau.get_skimmed_large_data(data_path, CENT_CLASSES, PT_BINS, CT_BINS, COLUMNS, application_columns, N_BODY, split)
            try:
                df_skimmed.to_parquet(os.path.dirname(data_path) + '/skimmed_df.parquet.gzip', compression='gzip')
            except:
                print("you need to run the training")
        ml_application = ModelApplication(N_BODY, data_path, analysis_res_path, CENT_CLASSES, split, df_skimmed)

    else:
        ml_application = ModelApplication(N_BODY, data_path, analysis_res_path, CENT_CLASSES, split)
    #get the histogram with the mass shift
    shift_hist = shift_file.Get("fit_mean"+split)
    #initialize the histogram with the mass pol0 fit
    hist_masses = []
    for bkgmodel in BKG_MODELS:
        hist = shift_hist.Clone(f"mass_{split}_{bkgmodel}")
        hist.SetTitle(f"mass_{split}_{bkgmodel}"+";#it{p}_{T} (GeV/#it{c});m (GeV/#it{c}^{})")
        hist_masses.append(hist)
    iBin = 0
    for cclass in CENT_CLASSES:
        cent_dir = results_file.mkdir(f'{cclass[0]}-{cclass[1]}{split}')
        df_applied = ml_application.df_data.query(standard_selection)
        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
            for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                mass_bins = 40 if ctbin[1] < 16 else 36
                sub_dir = cent_dir.mkdir(f'ct_{ctbin[0]}{ctbin[1]}') if 'ct' in FILE_PREFIX else cent_dir.mkdir(f'pt_{ptbin[0]}{ptbin[1]}')
                sub_dir.cd()
                if UNBINNED_FIT:
                    iBin = iBin + 1
                    iMod = 0
                    for bkgmodel in BKG_MODELS:
                        mass = ROOT.RooRealVar("m","m_{^{3}He+#pi}",2.97,3.015,"GeV/c^{2}")
                        width = ROOT.RooRealVar("width","B0 mass width",0.001,0.003,"GeV/c^2")
                        mb0 = ROOT.RooRealVar("mb0","B0 mass",2.989,2.993,"GeV^-1")
                        slope = ROOT.RooRealVar("slope","slope mass",-100.,0.,"GeV")
                        b0sig = ROOT.RooGaussian("b0sig","B0 sig PDF",mass,mb0,width)
                        c0 = ROOT.RooRealVar("c0","constant c0",-100.,100.,"GeV/c^{2}")
                        c1 = ROOT.RooRealVar("c1","constant c1",-100.,100.,"GeV/c^{2}")
                        c2 = ROOT.RooRealVar("c2","constant c2",-100.,100.,"GeV/c^{2}")
                        expo = ROOT.RooExponential("expo","expo for bkg",mass,slope)
                        pol1 = ROOT.RooPolynomial("pol1","pol1 for bkg",mass,ROOT.RooArgList(c0,c1))
                        pol2 = ROOT.RooPolynomial("pol2","pol2 for bkg",mass,ROOT.RooArgList(c0,c1,c2))

                        n1 = ROOT.RooRealVar("n1","n1 const",0.,10000,"GeV")  
                        n2 = ROOT.RooRealVar("n2","n2 const",0.,10000,"GeV")  
                        if bkgmodel == 'pol1':
                            func_list = ROOT.RooArgList(b0sig,pol1)
                        elif bkgmodel == 'pol2':
                            func_list = ROOT.RooArgList(b0sig,pol2)
                        else:
                            func_list = ROOT.RooArgList(b0sig,expo)

                        sum = ROOT.RooAddPdf("sum","b0sig+bkg",func_list,ROOT.RooArgList(n1,n2))

                        var = ROOT.RooRealVar("m","Example Variable",2.97,3.015)
                        roo_data = hau.df2roo(df_applied.query("ct<@ctbin[1] and ct>@ctbin[0] and pt<@ptbin[1] and pt>@ptbin[0]"), {'m': var})
                        sum.fitTo(roo_data)
                        xframe2 = ROOT.RooPlot()
                        xframe2.SetName(f'ct{ctbin[0]}{ctbin[1]}_pT{ptbin[0]}{ptbin[1]}_cen{cclass[0]}{cclass[1]}')
                        xframe2 = mass.frame(18)
                        roo_data.plotOn(xframe2)
                        sum.plotOn(xframe2)
                        xframe2.Write()
                        print(shift_hist.GetBinContent(iBin))
                        hist_masses[iMod].SetBinContent(iBin,mb0.getVal()-shift_hist.GetBinContent(iBin)/1000)
                        hist_masses[iMod].SetBinError(iBin,mb0.getError())
                        iMod = iMod + 1
                else:
                    mass_array = np.array(df_applied.query("ct<@ctbin[1] and ct>@ctbin[0] and pt<@ptbin[1] and pt>@ptbin[0]")['m'].values, dtype=np.float64)
                    counts, _ = np.histogram(mass_array, bins=mass_bins, range=[2.96, 3.05])
                    h1_minv = hau.h1_invmass(counts, cclass, ptbin, ctbin, bins=mass_bins, name="")

                    for bkgmodel in BKG_MODELS:
                        # create dirs for models
                        fit_dir = sub_dir.mkdir(bkgmodel)
                        fit_dir.cd()

                        #h1_minv.Write()
                        rawcounts, err_rawcounts, significance, err_significance, mu, mu_err, _, _ = hau.fit_hist(h1_minv, cclass, ptbin, ctbin, model=bkgmodel, fixsigma=params['SIGMA_MC'] , mode=N_BODY, split=split)

                        print("mu: ",mu*1000,"+-",mu_err*1000,"MeV/c^2")
                        print("B: ",1875.61294257+1115.683-(mu*1000),"+-",mu_err*1000,"MeV")



    results_file.cd()

    for iMod in range(0,len(BKG_MODELS)):
        hist_masses[iMod].Fit("pol0")
        hist_masses[iMod].GetYaxis().SetRangeUser(2.990, 2.994)
        hist_masses[iMod].Write()

print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')
