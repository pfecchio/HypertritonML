#!/usr/bin/env python3

import argparse
import math
import os
import random
from array import array
import uproot
import numpy as np
import yaml
from scipy import stats

from ROOT import (TF1, TH1D, TH2D, TAxis, TCanvas, TColor, TFile, TFrame, TIter, TKey,
                  TPaveText, gDirectory, gROOT, gStyle, gPad, AliPWGFunc)

gROOT.LoadMacro("/home/alidock/HypertritonML/Utils/YieldMean.C+")
from ROOT import YieldMean


kBlueC = TColor.GetColor('#1f78b4')
kBlueCT = TColor.GetColorTransparent(kBlueC, 0.5)
kRedC = TColor.GetColor('#e31a1c')
kRedCT = TColor.GetColorTransparent(kRedC, 0.5)
kPurpleC = TColor.GetColor('#911eb4')
kPurpleCT = TColor.GetColorTransparent(kPurpleC, 0.5)
kOrangeC = TColor.GetColor('#ff7f00')
kOrangeCT = TColor.GetColorTransparent(kOrangeC, 0.5)
kGreenC = TColor.GetColor('#33a02c')
kGreenCT = TColor.GetColorTransparent(kGreenC, 0.5)
kMagentaC = TColor.GetColor('#f032e6')
kMagentaCT = TColor.GetColorTransparent(kMagentaC, 0.5)
kYellowC = TColor.GetColor('#ffe119')
kYellowCT = TColor.GetColorTransparent(kYellowC, 0.5)
kBrownC = TColor.GetColor('#b15928')
kBrownCT = TColor.GetColorTransparent(kBrownC, 0.5)

random.seed(1989)

parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to the YAML configuration file")
args = parser.parse_args()

gROOT.SetBatch()
bw_file = TFile(os.environ['HYPERML_UTILS'] + '/BlastWaveFits.root')
bw = bw_file.Get('BlastWave/BlastWave0')
params=bw.GetParameters()
params[0]=2.991
pwg=AliPWGFunc()
bw=pwg.GetBGBW(params[0],params[1],params[2],params[3],params[4])
with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

var = 'p_{T}'
unit = 'GeV/#it{c}'

rangesFile = resultsSysDir + '/' + params['FILE_PREFIX'] + '_effranges.yaml'
with open(os.path.expandvars(rangesFile), 'r') as stream:
    try:
        ranges = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_dist.root'
distribution = TFile(file_name, 'recreate')

file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_results.root'
resultFile = TFile(file_name)
bkgModels = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']

for cclass in params['CENTRALITY_CLASS']:

    data_path = os.path.expandvars('$HYPERML_TABLES_2/DataTable.root')
    hist_centrality = uproot.open(data_path)['EventCounter']
    n_events = sum(hist_centrality[cclass[0]+1:cclass[1]])
    inDirName = f"{cclass[0]}-{cclass[1]}"
    resultFile.cd(inDirName)
    outDir = distribution.mkdir(inDirName)
    h2PreselEff = resultFile.Get(f'{inDirName}/SelEff')
    h1PreselEff = h2PreselEff.ProjectionX()

    for i in range(1, h1PreselEff.GetNbinsX()+1):
        h1PreselEff.SetBinError(i, 0)

    h1PreselEff.SetTitle(f';{var} ({unit}); Preselection efficiency')
    h1PreselEff.UseCurrentStyle()
    h1PreselEff.SetMinimum(0)


    hRawCounts = []
    raws = []
    errs = []

    for model in bkgModels:
        h1RawCounts = h1PreselEff.Clone(f"best_{model}")
        h1RawCounts.Reset()

        h2Significance = resultFile.Get(f'{inDirName}/significance_{model}')
        outDir.cd()
        h2Significance.ProjectionX().Write(f'significance_pt_{model}')

        for iBin in range(1, h1RawCounts.GetNbinsX()+1):

            h2RawCounts = resultFile.Get(f'{inDirName}/RawCounts{ranges["BEST"][iBin-1]}_{model}')
            h1RawCounts.SetBinContent(iBin, h2RawCounts.GetBinContent(
                iBin,1) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCounts.GetBinWidth(iBin))
            h1RawCounts.SetBinError(iBin, h2RawCounts.GetBinError(
                iBin,1) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCounts.GetBinWidth(iBin))
            raws.append([])
            errs.append([])

            for eff in np.arange(ranges['SCAN'][iBin-1][0], ranges['SCAN'][iBin-1][1], ranges['SCAN'][iBin-1][2]):
                h2RawCounts = resultFile.Get(f'{inDirName}/RawCounts{eff:g}_{model}')
                raws[iBin-1].append(h2RawCounts.GetBinContent(iBin,
                                                              1) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin)/n_events/0.25)
                errs[iBin-1].append(h2RawCounts.GetBinError(iBin,
                                                            1) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin)/n_events/0.25)

    h1PreselEff.Scale(0.5) #takes into account the rapidity correction
    h1RawCounts.UseCurrentStyle()
    h1RawCounts.Scale(1/n_events/0.25)
    tmpSyst = h1RawCounts.Clone("hSyst")
    if(cclass[1]==10):
        h1RawCounts.Fit(bw, "MI")
    
    for iBin in range(1, h1RawCounts.GetNbinsX() + 1):
        val = tmpSyst.GetBinContent(iBin)
        tmpSyst.SetBinError(iBin, np.std(raws[iBin - 1]))
    fout=TF1()
    res_hist=YieldMean(h1RawCounts,tmpSyst,fout,bw,0,20.,1e-3,1e-1,False,"log.root","","I")

    #final plots
    h1RawCounts.SetTitle(';p_{T} GeV/c;1/ (N_{ev}) d^{2}N/(dy dp_{T}) x B.R. (GeV/c)^{-1}')
    pinfo2= TPaveText(0.5,0.5,0.91,0.9,"NDC")
    pinfo2.SetBorderSize(0)
    pinfo2.SetFillStyle(0)
    pinfo2.SetTextAlign(30+3)
    pinfo2.SetTextFont(42)
    string ='ALICE Internal, Pb-Pb 2018 {}-{}%'.format(cclass[0],cclass[1])
    pinfo2.AddText(string)
    h1RawCounts.Draw()
    h1RawCounts.SetMarkerStyle(20)
    h1RawCounts.SetMarkerColor(600)
    h1RawCounts.SetLineColor(600)
    pinfo2.Draw("x0same")
    tmpSyst.SetFillStyle(0)

    outDir.cd()
    h1PreselEff.Write("h1PreselEff")
    res_hist.Write()

    myCv = TCanvas(f"ptSpectraCv_{model}")
    myCv.SetLogy()
    myCv.cd()
    h1RawCounts.Draw()
    tmpSyst.Draw("e2same")
    pinfo2.Draw("x0same")
    myCv.Write()
    myCv.Close()
    
resultFile.Close()
bw=-1
pwg=-1

