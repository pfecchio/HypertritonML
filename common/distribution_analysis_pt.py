#!/usr/bin/env python3

import analysis_utils as au

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
                  TPaveText, gDirectory, gROOT, gStyle, gPad, AliPWGFunc, kBlack, kBlue, kRed)

gROOT.LoadMacro("/home/alidock/HypertritonML/Utils/YieldMean.C+")

from ROOT import YieldMean
random.seed(1989)


parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to the YAML configuration file")
args = parser.parse_args()

gROOT.SetBatch()
bw_file = TFile(os.environ['HYPERML_UTILS'] + '/BlastWaveFits.root')
bw = bw_file.Get('BlastWave/BlastWave0')
params = bw.GetParameters()
params[0] = 2.992
pwg = AliPWGFunc()
bw = pwg.GetBGBW(params[0], params[1], params[2], params[3], params[4])
bw.SetParLimits(1, 0, 2)
bw.SetParLimits(2, 0, 1)
bw.SetParLimits(3, 0, 2)
with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

var = 'p_{T}'
unit = 'GeV/#it{c}'


file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_dist.root'
distribution = TFile(file_name, 'recreate')

split_list = ['_matter', '_antimatter']

absorption_list_values = [uproot.open(os.environ['HYPERML_UTILS'] + '/absorption.root')[
                                      'hOssHyp1'].values, uproot.open(os.environ['HYPERML_UTILS'] + "/absorption.root")['hOssAntiHyp1'].values]
absorption_list_edges = [uproot.open(os.environ['HYPERML_UTILS'] + '/absorption.root')[
                                     'hOssHyp1'].edges, uproot.open(os.environ['HYPERML_UTILS'] + "/absorption.root")['hOssAntiHyp1'].edges]

for cclass in params['CENTRALITY_CLASS']:

    hist_list = []
    eff_list = []
    syst_list = []
    inDirName = f"{cclass[0]}-{cclass[1]}"
    outDir = distribution.mkdir(inDirName)
    for split_string in split_list:

        rangesFile = resultsSysDir + '/' + \
            params['FILE_PREFIX'] + '_score_bdteff{}.yaml'.format(split_string)
        with open(os.path.expandvars(rangesFile), 'r') as stream:
            try:
                score_dict = yaml.full_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        ranges = au.create_ranges(score_dict)

        file_name = resultsSysDir + '/' + \
            params['FILE_PREFIX'] + '_results{}.root'.format(split_string)
        resultFile = TFile(file_name)
        bkgModels = params['BKG_MODELS'] if 'BKG_MODELS' in params else [
            'expo']

        data_path = os.path.expandvars('$HYPERML_TABLES_2/DataTable.root')
        hist_centrality = uproot.open(data_path)['EventCounter']
        n_events = sum(hist_centrality[cclass[0]+1:cclass[1]])
        
        resultFile.cd(inDirName)
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

            # h2Significance = resultFile.Get(f'{inDirName}/significance_{model}')
            # h2Significance.ProjectionX().Write(f'significance_pt_{model}')

            for iBin in range(1, h1RawCounts.GetNbinsX()+1):

                h2RawCounts = resultFile.Get(
                    f'{inDirName}/RawCounts{ranges["BEST"][iBin-1]}_{model}')
                h1RawCounts.SetBinContent(iBin, h2RawCounts.GetBinContent(
                    iBin, 1) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCounts.GetBinWidth(iBin))
                h1RawCounts.SetBinError(iBin, h2RawCounts.GetBinError(
                    iBin, 1) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCounts.GetBinWidth(iBin))
                raws.append([])
                errs.append([])
                for eff in np.arange(ranges['SCAN'][iBin-1][0], ranges['SCAN'][iBin-1][1], ranges['SCAN'][iBin-1][2]):
                    h2RawCounts = resultFile.Get(
                        f'{inDirName}/RawCounts{eff:g}_{model}')
                    raws[iBin-1].append(h2RawCounts.GetBinContent(iBin,
                                                                  1) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin)/n_events/0.25)
                    errs[iBin-1].append(h2RawCounts.GetBinError(iBin,
                                                                1) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin)/n_events/0.25)

        h1PreselEff.Scale(0.5)  
        h1RawCounts.UseCurrentStyle()
        h1RawCounts.Scale(1/n_events/0.25)
        abs_corr = h1RawCounts.Clone("abs_corr")
        abs_val = absorption_list_values[split_list.index(split_string)]
        abs_edg = absorption_list_edges[split_list.index(split_string)][1:]
        for iBin in range(1, h1RawCounts.GetNbinsX() + 1):
            low_edge = h1RawCounts.GetBinLowEdge(iBin)
            high_edge = h1RawCounts.GetBinLowEdge(iBin+1)
            absorption = np.mean(abs_val[np.logical_and(
                abs_edg >= low_edge, abs_edg < high_edge)])
            abs_corr.SetBinContent(iBin, absorption)

        h1RawCounts.Divide(abs_corr)
        tmpSyst = h1RawCounts.Clone("hSyst")
        for iBin in range(1, h1RawCounts.GetNbinsX() + 1):
            tmpSyst.SetBinError(iBin, np.std(raws[iBin - 1]))

        if(cclass[1] == 10):
            h1RawCounts.Fit(bw, "I")
            fout = TF1()
            res_hist = YieldMean(h1RawCounts, tmpSyst, fout, bw,
                                 0, 12., 1e-2, 1e-1, False, "log.root", "", "I")

        outDir.cd()                         

        h1RawCounts.SetTitle(
        ';p_{T} GeV/c;1/ (N_{ev}) d^{2}N/(dy dp_{T}) x B.R. (GeV/c)^{-1}')
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
        h1PreselEff.Write("h1PreselEff{}".format(split_string))
        if(cclass[1]==10):
            res_hist.Write("yield_mean{}".format(split_string))
        myCv = TCanvas("ptSpectraCv{}".format(split_string))
        myCv.SetLogy()
        myCv.cd()
        h1RawCounts.Draw()
        tmpSyst.Draw("e2same")
        pinfo2.Draw("x0same")
        myCv.Write()
        myCv.Close()
        hist_list.append(h1RawCounts.Clone("pT spectrum" + split_string))
        eff_list.append(h1PreselEff.Clone("Efficiecy x Acceptance" + split_string))
        syst_list.append(tmpSyst.Clone("cc"))



    
    myCv_common = TCanvas("ptSpectraCv_common")
    myCv_common.SetLogy()
    myCv_common.cd()
    hist_list[0].GetListOfFunctions().Remove(hist_list[0].GetFunction("fBGBW"))
    hist_list[1].GetListOfFunctions().Remove(hist_list[1].GetFunction("fBGBW"))
    hist_list[0].Draw()
    hist_list[1].SetMarkerColor(kRed)
    hist_list[1].SetLineColor(kRed)
    syst_list[1].SetLineColor(kRed)
    hist_list[1].Draw("same")
    syst_list[0].Draw("e2same")
    syst_list[1].Draw("e2same")
    pinfo2.Draw("x0same")
    myCv_common.Write()
    myCv_common.Close()

    myCv_sum = TCanvas("ptSpectraCv_sum")
    myCv_sum.SetLogy()
    myCv_sum.cd()
    hist_tot=hist_list[0].Clone("hist_sum")
    hist_tot.Add(hist_tot,hist_list[1])
    hist_tot.Draw()
    syst_tot=syst_list[0].Clone("syst_sum")
    syst_tot.Add(syst_tot,syst_list[1])
    syst_tot.Draw("e2same")

    pinfo2.Draw("x0same")
    myCv_sum.Write()
    myCv_sum.Close()

    cv_eff_common = TCanvas("EffCv_common")
    cv_eff_common.cd()
    eff_list[0].SetLineColor(kRed)
    eff_list[1].SetLineColor(kBlue)
    eff_list[0].Draw()
    eff_list[1].Draw("SAME")
    pinfo2.Draw("x0same")
    gPad.BuildLegend()
    cv_eff_common.Write()
    cv_eff_common.Close()
    cv_eff_common.Close()

    myCv_ratio = TCanvas("ratio")
    myCv_ratio.cd()
    hist_list[0].Divide(hist_list[1])
    syst_list[0].Divide(syst_list[1])
    hist_list[0].SetTitle(";p_{T} GeV/c;Matter/AntiMatter")
    hist_list[0].Draw("P")
    hist_list[0].SetMarkerColor(kBlue)
    hist_list[0].SetMarkerStyle(20)
    syst_list[0].Draw("e2same")


    pinfo2.Draw("x0same")
    myCv_ratio.Write()
    myCv_ratio.Close()


resultFile.Close()

bw = -1
pwg = -1
