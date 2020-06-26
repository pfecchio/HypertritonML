#!/usr/bin/env python3

import argparse
import math
import os
import random
from array import array
from multiprocessing import Pool

import numpy as np
import yaml

from ROOT import (TF1, TH1D, TH2D, TAxis, TCanvas, TColor, TFile, TFrame,
                  TIter, TKey, TPaveText, gDirectory, gPad, gROOT, gStyle,
                  kBlue, kRed)
import ROOT
from scipy import stats


random.seed(19434)

from ROOT import gROOT
gROOT.SetBatch(True)

gROOT.LoadMacro("GlobalChi2.h")
from ROOT import GlobalChi2

SPLIT_LIST = ['_matter', '_antimatter']

gROOT.SetBatch()

parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to the YAML configuration file")
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

var = '#it{ct}'
unit = 'cm'

file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_cpt.root'
distribution = TFile(file_name, 'recreate')

file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_results_fit.root'
results_file = TFile(file_name, 'read')

if(params["NBODY"] == 2):
    abs_file_name = os.environ['HYPERML_UTILS_{}'.format(
        params['NBODY'])] + '/he3abs/recCtHe3.root'
    absorp_file = TFile(abs_file_name)
    absorp_hist = absorp_file.Get('Reconstructed ct spectrum')

bkgModels = ["expo", "pol1","pol2"]


best_sig = [0.76, 0.79, 0.83, 0.81, 0.83, 0.83, 0.84, 0.75]
sig_ranges=[]
for i in best_sig:
    if i== best_sig[0]:
        sig_ranges.append([i-0.03, i+0.03, 0.01])
    else:
        sig_ranges.append([i-0.1, i+0.1, 0.01])

ranges = {
    'BEST': best_sig,
    'SCAN': sig_ranges
}

raw_list = []
err_list = []

out_dir = distribution.mkdir('0-90')

for split in SPLIT_LIST:
    for cclass in params['CENTRALITY_CLASS']:
        inDirName = f'{cclass[0]}-{cclass[1]}' + split
        results_file.cd(inDirName)

        h2PreselEff = results_file.Get(f'{inDirName}/PreselEff')
        h1PreselEff = h2PreselEff.ProjectionY("preseleff", 1, 1)

        for i in range(1, h1PreselEff.GetNbinsX() + 1):
            h1PreselEff.SetBinError(i, 0)

        h1PreselEff.SetTitle(f';{var} ({unit}); Preselection efficiency')
        h1PreselEff.UseCurrentStyle()
        h1PreselEff.SetMinimum(0)
        out_dir.cd()

        hRawCounts = []
        raws = []
        errs = []

        for model in bkgModels:
            h1RawCounts = h1PreselEff.Clone(f"best_{model}")
            h1RawCounts.Reset()

            out_dir.cd()

            for iBin in range(1, h1RawCounts.GetNbinsX() + 1):
                h2RawCounts = results_file.Get(f'{inDirName}/RawCounts{ranges["BEST"][iBin-1]:.2f}_{model}')
                raws.append([])
                errs.append([])

                abs_corr = 1
                if(params["NBODY"] == 2):
                    abs_corr = 1-absorp_hist.GetBinContent(iBin)

                for eff in np.arange(ranges['SCAN'][iBin - 1][0], ranges['SCAN'][iBin - 1][1], ranges['SCAN'][iBin - 1][2]):
                    if eff > 0.99:
                        continue

                    h2RawCounts = results_file.Get(f'{inDirName}/RawCounts{eff:.2f}_{model}')
                    raws[iBin-1].append(h2RawCounts.GetBinContent(1,
                                                                  iBin) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin)/abs_corr)
                    errs[iBin-1].append(h2RawCounts.GetBinError(1,
                                                                iBin) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin)/abs_corr)
    
    raw_list.append(raws)
    err_list.append(errs)





out_dir.cd()

syst_life = TH1D("syst", ";#tau (ps);Entries", 200, 230, 280)
syst_tau = TH1D("r", "; r ;Entries", 200, -0.15, 0.25)
syst_norm = TH1D("#delta_{norm}", "; #delta_{norm} ;Entries", 200, -400, 50)

chi2_histo = TH1D("global chi 2", "; chi2 ;Entries", 100, 0, 10)

                    # 300, 100, 400, 4000, 2500, 6500)
tmpCt_mat = h1PreselEff.Clone("hypertriton")
tmpCt_antimat = h1PreselEff.Clone("anti-hypertriton")
tmpCt_mat.SetTitle(";#it{c}t (cm);d#it{N}/d(#it{c}t) [(cm)^{-1}]")
tmpCt_antimat.SetTitle(";#it{c}t (cm);d#it{N}/d(#it{c}t) [(cm)^{-1}]")
tmpCt_mat.SetMinimum(1)
tmpCt_antimat.SetMinimum(1)
# Create observables
Expo1 = TF1(
    "myexpo", "[0]*exp(-x/([1]*0.029979245800))/([1]*0.029979245800)", 1, 35)
Expo1.SetParLimits(1, 100, 350)
Expo1.SetParLimits(0, 1000, 3000)
Expo1.SetParName(0, "Norm")
Expo1.SetParName(1, "#tau")
# Create observables
Expo2 = TF1(
    "myexpo2", "([0]+[3])*exp(-x/(([1]*(1+[2]/[1]))*0.029979245800))/(([1]*(1+[2]))*0.029979245800)", 1, 35)
Expo2.SetParLimits(1, 100, 350)
Expo2.SetParLimits(0, 1000, 3000)
Expo2.SetParName(0, "Norm")
Expo2.SetParName(1, "#tau")
Expo2.SetParName(2, "r")
Expo2.SetParName(3, "#delta norm")

combinations = set()
size = 10000
counter = 0
for _ in range(size):
    counter = counter + 1
    tmpCt_mat.Reset()
    tmpCt_antimat.Reset()
    comboList = []

    for iBin in range(1, tmpCt_mat.GetNbinsX() + 1):
        index = random.randint(0, len(raws[iBin-1])-1)
        comboList.append(index)
        tmpCt_mat.SetBinContent(iBin, raw_list[0][iBin-1][index])
        tmpCt_mat.SetBinError(iBin, err_list[0][iBin-1][index])
        tmpCt_antimat.SetBinContent(iBin, raw_list[1][iBin-1][index])
        tmpCt_antimat.SetBinError(iBin, err_list[1][iBin-1][index])





    # wf1 = ROOT.Math.WrappedMultiTF1(Expo1,1)
    # wf2 = ROOT.Math.WrappedMultiTF1(Expo2,1)

    # opt = ROOT.Fit.DataOptions()

    # data1 = ROOT.Fit.BinData(opt)
    # ROOT.Fit.FillData(data1, tmpCt_mat)

    # data2 = ROOT.Fit.BinData(opt)
    # ROOT.Fit.FillData(data2, tmpCt_antimat)

    # chi2_mat = ROOT.Fit.Chi2Function(data1, wf1)
    # chi2_anti = ROOT.Fit.Chi2Function(data2, wf2)

    globalChi2 = GlobalChi2(tmpCt_mat, tmpCt_antimat, Expo1, Expo2)

    fitter = ROOT.Fit.Fitter() 


    par0 = np.array([2000, 250, 10, 10], "double")

    fitter.Config().SetParamsSettings(4,par0)

    fitter.Config().ParSettings(1).SetLimits(100,350)


    fitter.Config().MinimizerOptions().SetPrintLevel(1)
    fitter.Config().SetMinimizer("Minuit2","Migrad")

    fitter.FitFCN(4, globalChi2)
    result = fitter.Result()
    # print("Result: ", result.MinFcnValue())




    combo = (x for x in comboList)
    if combo in combinations:
        continue
    
    if(result.MinFcnValue()/(8*2-4)<2):
        combinations.add(combo)
        syst_tau.Fill(result.Value(2))
        syst_norm.Fill(result.Value(3))
        syst_life.Fill(result.Value(1))
        chi2_histo.Fill(result.MinFcnValue()/(8*2-4))



c1 = TCanvas("Simfit","Simultaneous fit of two histograms", 10,10,700,700)

c1.Divide(1,2)
c1.cd(1)
gStyle.SetOptFit(1111)

Expo1.SetParameter(0, result.Value(0))
Expo1.SetParError(0, result.ParError(0))
Expo1.SetParameter(1, result.Value(1))
Expo1.SetParError(1, result.ParError(1))
# print(result.Chi2())
# print(result.Ndf())
Expo1.SetChisquare(result.MinFcnValue())
Expo1.SetNDF(8*2-4)

Expo1.SetLineColor(kBlue)
tmpCt_mat.GetListOfFunctions().Add(Expo1)
tmpCt_mat.Draw()

c1.cd(2)
Expo2.SetParameter(0, result.Value(0))
Expo2.SetParError(0, result.ParError(0))
Expo2.SetParameter(1, result.Value(1))
Expo2.SetParError(1, result.ParError(1))
Expo2.SetParameter(2, result.Value(2))
Expo2.SetParError(2, result.ParError(2))
Expo2.SetParameter(3, result.Value(3))
Expo2.SetParError(3, result.ParError(3))
Expo2.SetChisquare(result.MinFcnValue())
Expo2.SetNDF(8*2-4)


Expo2.SetLineColor(kBlue)
tmpCt_antimat.GetListOfFunctions().Add(Expo2)
tmpCt_antimat.Draw()
c1.Write()
    


syst_life.SetFillColor(600)
syst_life.SetFillStyle(3345)
syst_tau.SetFillColor(600)
syst_tau.SetFillStyle(3345)
syst_norm.SetFillColor(600)
syst_norm.SetFillStyle(3345)
chi2_histo.SetFillColor(600)
chi2_histo.SetFillStyle(3345)
# syst.Scale(1./syst.Integral())
syst_life.Write()
c2 = TCanvas("syst_tau_fit")
syst_tau.Fit("gaus")
syst_tau.Write()
syst_norm.Write()
chi2_histo.Write()



results_file.Close()
