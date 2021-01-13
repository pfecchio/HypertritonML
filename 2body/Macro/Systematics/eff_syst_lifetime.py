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
                  TPaveText, gDirectory, gROOT, gStyle, gPad, kBlue, kRed, kGreen, kOrange, TLegend)

kBlueC = TColor.GetColor('#1f78b4')


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


file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_pt_shape_syst.root'
distribution = TFile(file_name, 'recreate')

file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_results_fit.root'
# file_name = resultsSysDir + '/3b.root'
results_file = TFile(file_name, 'read')
ind_list = ["bw", "bol", "mtexp"]
ptshape_file_name = os.environ['HYPERML_UTILS_{}'.format(params['NBODY'])] + '/pt_shape_comparison.root'
ptshape_file = TFile(ptshape_file_name)
eff_hist_list = []
for ind in ind_list:
    eff_hist = ptshape_file.Get('PreselEff_' + ind)
    eff_hist.SetDirectory(0)
    eff_hist_list.append(eff_hist)


bkgModels = ["pol2"]
hist_list = []
out_dir = distribution.mkdir("eff")
eff_names_list = ["BlastWave", "Boltzmann", "Mt-Exponential"]
eff_colors_list = [kRed, kBlue, kGreen]
fit_func_list = []
for name, color, ind, eff_hist in zip(eff_names_list, eff_colors_list, ind_list, eff_hist_list):
    for cclass in params['CENTRALITY_CLASS']:
        inDirName = f"{cclass[0]}-{cclass[1]}"

        h2BDTEff = results_file.Get(f'{inDirName}/BDTeff')
        h1BDTEff = h2BDTEff.ProjectionY("bdteff", 1, 1)
        # best_sig = np.round(np.array(h1BDTEff)[1:-1], 2)
        # sig_ranges = []
        # for i in best_sig:
        #     if i == best_sig[0]:
        #         sig_ranges.append([i-0.03, i+0.03, 0.01])
        #     else:
        #         sig_ranges.append([i-0.1, i+0.1, 0.01])

        #            01    12    24    46    68    810   1014 1418  1823  2335
        best_sig = [0.25, 0.72, 0.80, 0.80, 0.83, 0.83, 0.81, 0.82, 0.80, 0.66]
        sig_ranges = [[0.22, 0.28, 0.01], [0.62, 0.82, 0.01], [0.74, 0.94, 0.01], [0.75, 0.95, 0.01], [0.75, 0.95, 0.01], [0.77, 0.97, 0.01], [0.71, 0.91, 0.01], [0.72, 0.92, 0.01], [0.70, 0.90, 0.01], [0.56, 0.76, 0.01]]

        ranges = {
            'BEST': best_sig,
            'SCAN': sig_ranges
        }

        h1PreselEff = eff_hist

        for i in range(1, h1PreselEff.GetNbinsX() + 1):
            h1PreselEff.SetBinError(i, 0)

        hRawCounts = []
        raws = []
        errs = []

        for model in bkgModels:
            h1RawCounts = h1PreselEff.Clone(f"best_{model}")
            h1RawCounts.Reset()

            for iBin in range(1, h1RawCounts.GetNbinsX()+1):
                h2RawCounts = results_file.Get(f'{inDirName}/RawCounts{ranges["BEST"][iBin-1]:.2f}_{model}')
                h1RawCounts.SetBinContent(iBin, h2RawCounts.GetBinContent(1, iBin) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCounts.GetBinWidth(iBin))
                h1RawCounts.SetBinError(iBin, h2RawCounts.GetBinError(1, iBin) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCounts.GetBinWidth(iBin))
                raws.append([])
                errs.append([])

                for eff in np.arange(ranges['SCAN'][iBin-1][0], ranges['SCAN'][iBin-1][1], ranges['SCAN'][iBin-1][2]):
                    h2RawCounts = results_file.Get(f'{inDirName}/RawCounts{eff:.2f}_{model}')
                    raws[iBin-1].append(h2RawCounts.GetBinContent(1, iBin) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin))
                    errs[iBin-1].append(h2RawCounts.GetBinError(1, iBin) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin))

            out_dir.cd()
            h1RawCounts.UseCurrentStyle()
            h1RawCounts.Fit(expo, "MI0+", "", 0, 35)
            fit_function = h1RawCounts.GetFunction("myexpo")
            fit_function.SetLineColor(color)
            # h1RawCounts.Write()
            # hRawCounts.append(h1RawCounts)
            if model == "pol2":
                fit_func_list.append(fit_function.Clone(name))
                myCv = TCanvas(f"ctSpectraCv_{model}_eff_{ind}")
                myCv.SetLogy()
                frame = gPad.DrawFrame(
                    0., 3, 36, 2000, ";#it{c}t (cm);d#it{N}/d(#it{c}t) [(cm)^{-1}]")
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
                myCv.Write()

absCv = TCanvas(f"ptshape_lifetimes")
absCv.SetLogy()
frame = gPad.DrawFrame(
    0., 3, 36, 2000, ";#it{c}t (cm);d#it{N}/d(#it{c}t) [(cm)^{-1}]")
frame.GetYaxis().SetTitleSize(26)
frame.GetYaxis().SetLabelSize(22)
frame.GetXaxis().SetTitleSize(26)
frame.GetXaxis().SetLabelSize(22)
# gStyle.SetLegendTextSize(30.)
# gStyle.SetLegendFont(42)
leg = TLegend(0.43, 0.65, 0.9, 0.87)
for name, func in zip(eff_names_list, fit_func_list):
    string = '#tau = {:.0f} #pm {:.0f} ps '.format(
        func.GetParameter(1), func.GetParError(1))
    func.SetTitle(name + " :  " + string)
    func.Draw("same")
    leg.AddEntry(func)
leg.SetHeader("Pt-shape model")

leg.Draw()
absCv.Write()


absCv = TCanvas("efficiency_corrections")
frame = gPad.DrawFrame(0.14, 0, 35, 0.3, ";#it{c}t (cm); Efficiency")
frame.GetYaxis().SetTitleSize(26)
frame.GetYaxis().SetLabelSize(22)
frame.GetXaxis().SetTitleSize(26)
frame.GetXaxis().SetLabelSize(22)
leg = TLegend(0.4, 0.6, 0.9, 0.85)
for color, name, hist in zip(eff_colors_list, eff_names_list, eff_hist_list):
    # for iBin in range(1, h1RawCounts.GetNbinsX()+1):
    #     hist.SetBinError(iBin, 0)
    hist.SetTitle(name)
    hist.SetMarkerSize(0)
    hist.SetLineColor(color)
    hist.Draw("hist same")
    leg.AddEntry(hist)
leg.SetHeader("Pt-shape model")
leg.Draw("hist")
absCv.Write()

results_file.Close()
