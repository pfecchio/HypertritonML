#!/usr/bin/env python3

import hyp_analysis_utils as au
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
                  TPaveText, gDirectory, gROOT, gStyle, gPad, AliPWGFunc, kBlack, kBlue, kRed, kOrange, kGreen, TLegend)


gROOT.LoadMacro("../../../Utils/YieldMean.C")

from ROOT import yieldmean
random.seed(1989)


parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to the YAML configuration file")
args = parser.parse_args()

gROOT.SetBatch()
bw_file = TFile(os.environ['HYPERML_UTILS'] + '/BlastWaveFits.root')


bw = bw_file.Get('BlastWave/BlastWave0')
params = bw.GetParameters()
params[0] = 2.991
pwg = AliPWGFunc()
bw = pwg.GetBGBW(params[0], params[1], params[2], params[3], params[4])
bw.SetParLimits(1, 0, 2)
bw.SetParLimits(2, 0, 1)
bw.SetParLimits(3, 0, 2)
bw.SetName("pt_func")
bw.SetTitle("pt_func")


with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

var = 'p_{T}'
unit = 'GeV/#it{c}'


split_list = ['_matter', '_antimatter']


file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_abs.root'
distribution = TFile(file_name, 'recreate')

file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_results_fit.root'

results_file = TFile(file_name, 'read')

analysis_res_path = os.path.expandvars(params['ANALYSIS_RESULTS_PATH'])

# if(params["NBODY"]==2):
#     abs_file_name = os.environ['HYPERML_UTILS_{}'.format(params['NBODY'])] + '/he3abs/absorption_pt/recPtHe3.root'
#     absorp_file = TFile(abs_file_name)
#     absorp_hist = absorp_file.Get('Reconstructed ct spectrum')

bkgModels = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']


ind_list = ["", "_1.5", "_2", "_10"]
abs_hist_list = []
for ind in ind_list:
    abs_file_name = os.environ['HYPERML_UTILS_{}'.format(params['NBODY'])] + '/he3abs/absorption_pt/recPtHe3' + ind + ".root"
    absorp_file = TFile(abs_file_name)
    abs_hist = absorp_file.Get('Reconstructed pT spectrum')
    abs_hist.SetDirectory(0)
    abs_hist_list.append(abs_hist)



hist_centrality = uproot.open(analysis_res_path)["AliAnalysisTaskHyperTriton2He3piML_custom_summary"][11]


for split in split_list:
    
    abs_names_list = ["100%", "150%", "200%", "1000%"]
    abs_colors_list = [kRed, kBlue, kGreen, kOrange]
    fit_func_list = []

    cclass = params['CENTRALITY_CLASS'][0]

    n_events = sum(hist_centrality[cclass[0]+1:cclass[1]])
    inDirName = f'{cclass[0]}-{cclass[1]}' + split

    h2BDTEff = results_file.Get(f'{inDirName}/BDTeff')
    h1BDTEff = h2BDTEff.ProjectionX("bdteff")

    best_sig = np.round(np.array(h1BDTEff)[1:-1], 2)
    sig_ranges = []
    for i in best_sig:
        sig_ranges.append([i-0.03, i+0.03, 0.01])

    ranges = {
            'BEST': best_sig,
            'SCAN': sig_ranges
    }

    results_file.cd(inDirName)
    out_dir = distribution.mkdir(inDirName)

    h2PreselEff = results_file.Get(f'{inDirName}/PreselEff')
    h1PreselEff = h2PreselEff.ProjectionX("preseleff")

    for i in range(1, h1PreselEff.GetNbinsX() + 1):
        h1PreselEff.SetBinError(i, 0)

    h1PreselEff.SetTitle(f';{var} ({unit}); Preselection efficiency')
    h1PreselEff.UseCurrentStyle()
    h1PreselEff.SetMinimum(0)
    out_dir.cd()

    for name, color,ind,absorp_hist in zip(abs_names_list,abs_colors_list, ind_list, abs_hist_list):

        hRawCounts = []
        raws = []
        errs = []

        for model in bkgModels:
            h1RawCounts = h1PreselEff.Clone(f"best_{model}")
            h1RawCounts.Reset()


            for iBin in range(1, h1RawCounts.GetNbinsX()+1):
                h2RawCounts = results_file.Get(f'{inDirName}/RawCounts{ranges["BEST"][iBin-1]:.2f}_{model}')


                h1RawCounts.SetBinContent(iBin, h2RawCounts.GetBinContent(
                    iBin, 1) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCounts.GetBinWidth(iBin)/(1-absorp_hist.GetBinContent(iBin)))
                h1RawCounts.SetBinError(iBin, h2RawCounts.GetBinError(
                    iBin, 1) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCounts.GetBinWidth(iBin)/(1-absorp_hist.GetBinContent(iBin)))
                raws.append([])
                errs.append([])

                for eff in np.arange(ranges['SCAN'][iBin-1][0], ranges['SCAN'][iBin-1][1], ranges['SCAN'][iBin-1][2]):
                    h2RawCounts = results_file.Get(f'{inDirName}/RawCounts{eff:.2f}_{model}')
                    raws[iBin-1].append(h2RawCounts.GetBinContent(iBin,1) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin)/n_events/(1-absorp_hist.GetBinContent(iBin)))
                    errs[iBin-1].append(h2RawCounts.GetBinError(iBin,1) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin)/n_events/(1-absorp_hist.GetBinContent(iBin)))


        h1RawCounts.UseCurrentStyle()
        h1RawCounts.Scale(1/n_events)
        tmpSyst = h1RawCounts.Clone("hSyst")
        for iBin in range(1, h1RawCounts.GetNbinsX() + 1):
            tmpSyst.SetBinError(iBin, stats.median_absolute_deviation(raws[iBin - 1]))


##------------------ Fill YieldMean histo-----------------------------------------
        if(cclass[1] == 10):
            print(f"ABSORPTION PERCENTAGE: {name}, SPLIT: {split}")
            print("--------------------------------------")
            h1RawCounts.Fit(bw, "I")
            fit_function  = h1RawCounts.GetFunction("pt_func")
            fit_function.SetLineColor(color)
            fit_func_list.append(fit_function)

            fout = TF1()
            res_hist = yieldmean.YieldMean(h1RawCounts, tmpSyst, fout, bw,
                                0, 12., 1e-2, 1e-1, False, "log.root", "", "I")
            res_hist.SetName(res_hist.GetName() + name)
            res_hist.SetTitle(res_hist.GetTitle() + name)

        out_dir.cd()

        if(cclass[1] == 10):
            res_hist.Write()
##--------------------------------------------------------------------------------



##------------------Fill corrected spectrum histo---------------------------------
        h1RawCounts.SetTitle(';p_{T} GeV/c;1/ (N_{ev}) d^{2}N/(dy dp_{T}) x B.R. (GeV/c)^{-1}')
        h1RawCounts.SetTitle("spectrum_" + name)
        pinfo2 = TPaveText(0.5,0.5,0.91,0.9,"NDC")
        pinfo2.SetBorderSize(0)
        pinfo2.SetFillStyle(0)
        pinfo2.SetTextAlign(30+3)
        pinfo2.SetTextFont(42)
        string = 'ALICE Internal, Pb-Pb 2018 {}-{}%'.format(cclass[0],cclass[1])
        pinfo2.AddText(string)
        h1RawCounts.SetMarkerStyle(20)
        h1RawCounts.SetMarkerColor(kBlue)
        h1RawCounts.SetLineColor(600)
        tmpSyst.SetFillStyle(0)
        myCv = TCanvas("ptSpectraCv{}_{}".format(split, name))
        myCv.SetLogy()
        myCv.cd()
        h1RawCounts.Draw()
        tmpSyst.Draw("e2same")
        pinfo2.Draw("x0same")
        myCv.Write()
        myCv.Close()
##----------------------------------------------------------------------------------



absCv = TCanvas("absorption_corrections")
frame = gPad.DrawFrame(2., 0, 9, 0.17, ";#it{p}_{T} (GeV/#it{c}); P_{abs}")
frame.GetYaxis().SetTitleSize(26)
frame.GetYaxis().SetLabelSize(22)
frame.GetXaxis().SetTitleSize(26)
frame.GetXaxis().SetLabelSize(22)
leg = TLegend(0.4,0.6,0.9,0.85)
for color, name, hist in zip(abs_colors_list, abs_names_list, abs_hist_list):
    # for iBin in range(1, h1RawCounts.GetNbinsX()+1):
    #     hist.SetBinError(iBin, 0)
    hist.SetTitle(name)
    hist.SetMarkerSize(0)
    hist.SetLineColor(color)
    hist.Draw("hist same")
    leg.AddEntry(hist)
leg.SetHeader("Percentage of anti - ^{3}He inelastic cross-section")
leg.Draw("hist")
absCv.Write()






results_file.Close()

bw = -1
pwg = -1
