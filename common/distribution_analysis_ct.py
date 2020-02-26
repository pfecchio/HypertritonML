#!/usr/bin/env python3

import argparse
import math
import os
import random
from array import array

import numpy as np
import yaml
from scipy import stats

from ROOT import (TF1, TH1D, TH2D, TAxis, TCanvas, TColor, TFile, TFrame, TIter, TKey,
                  TPaveText, gDirectory, gROOT, gStyle, gPad)

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

expo = TF1(
    "myexpo", "[0]*exp(-x/([1]*0.029979245800))/([1]*0.029979245800)", 0, 35)
expo.SetParLimits(1, 100, 350)

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

var = '#it{ct}'
unit = 'cm'


file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_dist.root'
distribution = TFile(file_name, 'recreate')

file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_results_fit.root'
# file_name = resultsSysDir + '/3b.root'
results_file = TFile(file_name, 'read')




# file_name = '~/HypertritonML/3body/PreselectionEfficiency/PreselectionEfficiencyHist.root'
# eff_file = TFile(file_name, 'read')

bkgModels = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']

for cclass in params['CENTRALITY_CLASS']:
    inDirName = f"{cclass[0]}-{cclass[1]}"

    h2BDTEff = results_file.Get(f'{inDirName}/BDTeff')
    h1BDTEff = h2BDTEff.ProjectionY("bdteff", 1, 1)
    best_sig = np.round(np.array(h1BDTEff)[1:-1], 2)
    sig_ranges = []
    for i in best_sig:
        sig_ranges.append([i-0.1, i+0.1, 0.01])
        ranges = {
            'BEST': best_sig,
            'SCAN': sig_ranges
        }
    results_file.cd(inDirName)
    out_dir = distribution.mkdir(inDirName)
    cvDir = out_dir.mkdir("canvas")

    h2PreselEff = results_file.Get(f'{inDirName}/PreselEff')
    h1PreselEff = h2PreselEff.ProjectionY("preseleff", 1, 1)

    # h1PreselEff = eff_file.Get('fHistEfficiencyVsCt')

    for i in range(1, h1PreselEff.GetNbinsX() + 1):
        h1PreselEff.SetBinError(i, 0)

    h1PreselEff.SetTitle(f';{var} ({unit}); Preselection efficiency')
    h1PreselEff.UseCurrentStyle()
    h1PreselEff.SetMinimum(0)
    out_dir.cd()
    h1PreselEff.Write("h1PreselEff")

    hRawCounts = []
    raws = []
    errs = []

    for model in bkgModels:
        h1RawCounts = h1PreselEff.Clone(f"best_{model}")
        h1RawCounts.Reset()

        # h2Significance = results_file.Get(f'{inDirName}/significance_{model}')
        out_dir.cd()
        # h2Significance.ProjectionY().Write(f'significance_ct_{model}')

        for iBin in range(1, h1RawCounts.GetNbinsX()+1):
            h2RawCounts = results_file.Get(f'{inDirName}/RawCounts{ranges["BEST"][iBin-1]:.2f}_{model}')
            h1RawCounts.SetBinContent(iBin, h2RawCounts.GetBinContent(
                1, iBin) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCounts.GetBinWidth(iBin))
            h1RawCounts.SetBinError(iBin, h2RawCounts.GetBinError(
                1, iBin) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCounts.GetBinWidth(iBin))
            raws.append([])
            errs.append([])

            for eff in np.arange(ranges['SCAN'][iBin-1][0], ranges['SCAN'][iBin-1][1], ranges['SCAN'][iBin-1][2]):
                h2RawCounts = results_file.Get(f'{inDirName}/RawCounts{eff:.2f}_{model}')
                raws[iBin-1].append(h2RawCounts.GetBinContent(1,
                                                              iBin) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin))
                errs[iBin-1].append(h2RawCounts.GetBinError(1,
                                                            iBin) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin))

        out_dir.cd()
        h1RawCounts.UseCurrentStyle()
        h1RawCounts.Fit(expo, "MI0+")
        fit_function = h1RawCounts.GetFunction("myexpo")
        fit_function.SetLineColor(kOrangeC)
        h1RawCounts.Write()
        hRawCounts.append(h1RawCounts)

        cvDir.cd()
        myCv = TCanvas(f"ctSpectraCv_{model}")
        myCv.SetLogy()
        frame = gPad.DrawFrame(
            0., 3, 36, 2000, ";#it{c}t (cm);d#it{N}/d(#it{c}t) [(cm)^{-1}]")
        pinfo2 = TPaveText(0.5, 0.65, 0.88, 0.86, "NDC")
        pinfo2.SetBorderSize(0)
        pinfo2.SetFillStyle(0)
        pinfo2.SetTextAlign(22)
        pinfo2.SetTextFont(43)
        pinfo2.SetTextSize(22)
        string1 = '#bf{ALICE Preliminary}'
        string2 = 'Pb-Pb  #sqrt{#it{s}_{NN}} = 5.02 TeV,  0-90%'
        pinfo2.AddText(string1)
        pinfo2.AddText(string2)
        string = '#tau = {:.0f} #pm {:.0f} ps '.format(
            expo.GetParameter(1), expo.GetParError(1))
        pinfo2.AddText(string)
        if expo.GetNDF()is not 0:
            string = f'#chi^{{2}} / NDF = {(expo.GetChisquare() / expo.GetNDF()):.2f}'
        pinfo2.AddText(string)
        fit_function.Draw("same")
        h1RawCounts.Draw("ex0same")
        h1RawCounts.SetMarkerStyle(20)
        h1RawCounts.SetMarkerColor(kBlueC)
        h1RawCounts.SetLineColor(kBlueC)
        h1RawCounts.SetMinimum(0.001)
        h1RawCounts.SetMaximum(1000)
        frame.GetYaxis().SetTitleSize(26)
        frame.GetYaxis().SetLabelSize(22)
        frame.GetXaxis().SetTitleSize(26)
        frame.GetXaxis().SetLabelSize(22)
        h1RawCounts.SetStats(0)

        pinfo2.Draw("x0same")
        tmpSyst = h1RawCounts.Clone("hSyst")
        corSyst = h1RawCounts.Clone("hCorr")
        tmpSyst.SetFillStyle(0)
        tmpSyst.SetMinimum(0.001)
        tmpSyst.SetMaximum(1000)
        corSyst.SetFillStyle(3345)
        for iBin in range(1, h1RawCounts.GetNbinsX() + 1):
            print(iBin)
            print(len(raws))
            tmpSyst.SetBinError(iBin, np.std(raws[iBin - 1]))
            # corSyst.SetBinError(iBin, 0.086 * val)
        tmpSyst.SetLineColor(kBlueC)
        tmpSyst.SetMarkerColor(kBlueC)
        tmpSyst.Draw("e2same")
        # corSyst.Draw("e2same")
        out_dir.cd()
        myCv.Write()

        h1RawCounts.Draw("ex0same")
        pinfo2.Draw()
        cvDir.cd()
        myCv.Write()

    out_dir.cd()

    syst = TH1D("syst", ";#tau (ps);Entries", 300, 100, 400)
    prob = TH1D("prob", ";Lifetime fit probability;Entries", 100, 0, 1)
    pars = TH2D("pars", ";#tau (ps);Normalisation;Entries", 300, 100, 400, 4000, 2500, 6500)
    tmpCt = hRawCounts[0].Clone("tmpCt")

    combinations = set()
    size = 10000

    for _ in range(size):
        tmpCt.Reset()
        comboList = []

        for iBin in range(1, tmpCt.GetNbinsX()):
            index = random.randint(0, len(raws[iBin-1])-1)
            comboList.append(index)
            tmpCt.SetBinContent(iBin, raws[iBin-1][index])
            tmpCt.SetBinError(iBin, errs[iBin-1][index])

        combo = (x for x in comboList)
        if combo in combinations:
            continue

        combinations.add(combo)
        tmpCt.Fit(expo, "MI")
        prob.Fill(expo.GetProb())
        if expo.GetChisquare() < 3 * expo.GetNDF():
            syst.Fill(expo.GetParameter(1))
            pars.Fill(expo.GetParameter(1), expo.GetParameter(0))

    syst.SetFillColor(600)
    syst.SetFillStyle(3345)
    syst.Scale(1./syst.Integral())
    syst.Write()
    prob.Write()
    pars.Write()

results_file.Close()
