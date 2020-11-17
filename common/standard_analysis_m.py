#!/usr/bin/env python3
import argparse
import os
import time
import warnings
from array import array
import math

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
ROOT.ROOT.EnableImplicitMT()
ROOT.RooMsgService.instance().setSilentMode(True)

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
PASS = os.getenv("HYPERML_PASS")
OTF = os.getenv("HYPERML_OTF")

SPLIT_CUTS = ['']
SPLIT_LIST = ['']
if SPLIT_MODE:
    SPLIT_LIST = ['_matter','_antimatter']
    SPLIT_CUTS = ['&& ArmenterosAlpha > 0', '&& ArmenterosAlpha < 0']

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

file_name = results_dir + f'/{FILE_PREFIX}_std_results_pass{PASS}{OTF}.root'
results_file = TFile(file_name, 'recreate')

standard_selection = 'V0CosPA > 0.9999 && NpidClustersHe3 > 80 && He3ProngPt > 1.8 && pt > 2 && pt < 10 && PiProngPt > 0.15 && He3ProngPvDCA > 0.05 && PiProngPvDCA > 0.2 && TPCnSigmaHe3 < 3.5 && TPCnSigmaHe3 > -3.5 && ProngsDCA < 1'


rdfData = ROOT.RDataFrame("DataTable",data_path)
rdfMC = ROOT.RDataFrame("SignalTable",signal_path)

mass = ROOT.RooRealVar("m","m_{^{3}He+#pi}",2.975, 3.01,"GeV/c^{2}")
width = ROOT.RooRealVar("width","B0 mass width",0.001,0.002,"GeV/c^2")
mb0 = ROOT.RooRealVar("mb0","B0 mass",2.989,2.993,"GeV^-1")
slope = ROOT.RooRealVar("slope","slope mass",-100.,100,"GeV")
b0sig = ROOT.RooGaussian("b0sig","B0 sig PDF",mass,mb0,width)
c0 = ROOT.RooRealVar("c0","constant c0",-100.,100.,"GeV/c^{2}")
c1 = ROOT.RooRealVar("c1","constant c1",-100.,100.,"GeV/c^{2}")
c2 = ROOT.RooRealVar("c2","constant c2",-100.,100.,"GeV/c^{2}")
expo = ROOT.RooExponential("expo","expo for bkg",mass,slope)
pol1 = ROOT.RooPolynomial("pol1","pol1 for bkg",mass,ROOT.RooArgList(c0,c1))
pol2 = ROOT.RooPolynomial("pol2","pol2 for bkg",mass,ROOT.RooArgList(c0,c1,c2))
n1 = ROOT.RooRealVar("n1","n1 const",0.,1.,"GeV")
width2 = ROOT.RooRealVar("tails width","tails width",0.002,0.006,"GeV")
resSig = ROOT.RooGaussian("resSig", "signal + resolution", mass, mb0, width2)
convSig = ROOT.RooAddPdf("convSig","Double Gaussian",ROOT.RooArgList(b0sig, resSig),ROOT.RooArgList(n1))

bLfunction = ROOT.TF1("bLfunction", "1115.683 + 1875.61294257 - [0]",0,10)
rooFunList = []
for bkgmodel in BKG_MODELS:
    if bkgmodel == 'pol1':
        func_list = ROOT.RooArgList(b0sig,pol1)
    elif bkgmodel == 'pol2':
        func_list = ROOT.RooArgList(b0sig,pol2)
    else:
        func_list = ROOT.RooArgList(b0sig,expo)
    rooFunList.append(ROOT.RooAddPdf(f"{bkgmodel}_gaus","b0sig+bkg",func_list,ROOT.RooArgList(n1)))


for split,splitcut in zip(SPLIT_LIST,SPLIT_CUTS):
    NBINS = len(CT_BINS) - 1 if 'ct' in FILE_PREFIX else len(PT_BINS) - 1
    BINS = np.asarray(CT_BINS if 'ct' in FILE_PREFIX else PT_BINS, dtype=np.float64) 
    hist_massesData = []
    hist_massesMC = [
        ROOT.TH1D(f"massMC_{split}_gauss",";#it{p}_{T} (GeV/#it{c});#it{c}t (cm);m (MeV/#it{c}^{2})", NBINS, BINS),
        ROOT.TH1D(f"massMC_{split}_moment",";#it{p}_{T} (GeV/#it{c});#it{c}t (cm);m (MeV/#it{c}^{2})", NBINS, BINS)
    ]
    for bkgmodel in BKG_MODELS:
        hist_massesData.append(ROOT.TH1D(f"massData_{split}_{bkgmodel}",";#it{p}_{T} (GeV/#it{c});#it{c}t (cm);m (MeV/#it{c}^{2})", NBINS, BINS))

    dfData_applied = rdfData.Filter(standard_selection + splitcut)
    dfMC_applied = rdfMC.Filter(standard_selection + splitcut)

    for cclass in CENT_CLASSES:
        cent_dir = results_file.mkdir(f'{cclass[0]}-{cclass[1]}{split}')
        dfData_cent = dfData_applied.Filter(f"centrality >= {cclass[0]} && centrality < {cclass[1]}")
        dfMC_cent = dfMC_applied.Filter(f"centrality >= {cclass[0]} && centrality < {cclass[1]}")
        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
            pt = 0.5 * (ptbin[0] + ptbin[1])
            for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                ct = 0.5 * (ctbin[0] + ctbin[1])
                binNo = hist_massesMC[0].GetXaxis().FindBin(ct) if 'ct' in FILE_PREFIX else hist_massesMC[0].GetXaxis().FindBin(pt)
                
                mass_bins = 35
                sub_dir = cent_dir.mkdir(f'ct_{ctbin[0]}{ctbin[1]}') if 'ct' in FILE_PREFIX else cent_dir.mkdir(f'pt_{ptbin[0]}{ptbin[1]}')
                sub_dir.cd()
                massData_array = dfData_cent.Filter(f"ct<{ctbin[1]} && ct>{ctbin[0]} && pt<{ptbin[1]} && pt>{ptbin[0]}").AsNumpy(["m"])
                massMC_array = dfMC_cent.Filter(f"ct<{ctbin[1]} && ct>{ctbin[0]} && pt<{ptbin[1]} && pt>{ptbin[0]}").AsNumpy(["m"])
                if UNBINNED_FIT:
                    roo_data = hau.ndarray2roo(massData_array['m'], mass)
                    roo_mc = hau.ndarray2roo(massMC_array['m'], mass)
                else:
                    countsData = np.histogram(massData_array['m'], mass_bins, range=[2.975, 3.01])
                    countsMC = np.histogram(massMC_array['m'], mass_bins, range=[2.975, 3.01])
                    h1_minvData = hau.h1_invmass(countsData, cclass, ptbin, ctbin, name="data")
                    h1_minvMC = hau.h1_invmass(countsMC, cclass, ptbin, ctbin, name="MC")
                    roo_data = ROOT.RooDataHist(f'Roo{h1_minvData.GetName()}', 'Data', ROOT.RooArgList(mass), h1_minvData)
                    roo_mc = ROOT.RooDataHist(f'Roo{h1_minvMC.GetName()}', 'MC', ROOT.RooArgList(mass), h1_minvMC)
                
                convSig.fitTo(roo_mc)
                xframe = mass.frame(mass_bins)
                xframe.SetName(f'frameMC_ct{ctbin[0]}{ctbin[1]}_pT{ptbin[0]}{ptbin[1]}_cen{cclass[0]}{cclass[1]}')
                roo_mc.plotOn(xframe)
                convSig.plotOn(xframe)
                convSig.paramOn(xframe)
                hist_massesMC[0].SetBinContent(binNo, mb0.getVal() * 1000 - 2991.31)
                hist_massesMC[0].SetBinError(binNo, mb0.getError() * 1000)
                hist_massesMC[1].SetBinContent(binNo, roo_mc.mean(mass) * 1000 - 2991.31)
                hist_massesMC[1].SetBinError(binNo, roo_mc.moment(mass, 2) * 1000)
                xframe.Write()
                for pdf, dataHist in zip(rooFunList, hist_massesData):
                    pdf.fitTo(roo_data)
                    xframe = mass.frame(mass_bins)
                    xframe.SetName(f'{pdf.GetName()}_frameData_ct{ctbin[0]}{ctbin[1]}_pT{ptbin[0]}{ptbin[1]}_cen{cclass[0]}{cclass[1]}')
                    roo_data.plotOn(xframe)
                    pdf.plotOn(xframe)
                    dataHist.SetBinContent(binNo, mb0.getVal() * 1000)
                    dataHist.SetBinError(binNo, mb0.getError() * 1000)
                    xframe.Write()

    results_file.cd()
    for h in hist_massesData:
        mass_gauss = h.Clone()
        mass_gauss.Add(hist_massesMC[0], -1)
        print(mass_gauss.GetName())
        mass_gauss.Fit(bLfunction)
        mass_gauss.Write(f"{h.GetName()}_gauss")
        mass_moment = h.Clone()
        mass_moment.Add(hist_massesMC[1], -1)
        mass_moment.Fit(bLfunction)
        mass_moment.Write(f"{h.GetName()}_moment")
        h.Write()
    hist_massesMC[0].Write()
    hist_massesMC[1].Write()


print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')
