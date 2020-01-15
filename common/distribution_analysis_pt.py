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
from statsmodels.robust.scale import huber

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
    ranges_list=[]
    raws_list=[]
    hist_list = []
    eff_list = []
    syst_list = []
    inDirName = f"{cclass[0]}-{cclass[1]}"
    outDir = distribution.mkdir(inDirName)
    for split_string in split_list:
        file_name = resultsSysDir + '/' + \
            params['FILE_PREFIX'] + '_results{}.root'.format(split_string)
        resultFile = TFile(file_name)
        bkgModels = params['BKG_MODELS'] if 'BKG_MODELS' in params else [
            'expo']
        resultFile.cd(inDirName)
        h2BDTEff = resultFile.Get(f'{inDirName}/BDTeff')
        h1BDTEff = h2BDTEff.ProjectionX()

        best_sig = np.round(np.array(h1BDTEff)[1:-1],2)
        sig_ranges=[]
        for i in best_sig:
            sig_ranges.append([i-0.1,i+0.1,0.01])
        ranges={
        'BEST' : best_sig,
        'SCAN' : sig_ranges
        }
        
        data_path = os.path.expandvars(params['DATA_PATH'])
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
                    val = h2RawCounts.GetBinContent(iBin,1) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin)/n_events/0.25
                    raws[iBin-1].append(val)
                    errs[iBin-1].append(h2RawCounts.GetBinError(iBin,
                                                                1) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin)/n_events/0.25)
                    # if(split_string=='_matter' and iBin==1 and val == 1.1343769943680133e-07):
                    #     print('raws:' ,raws[iBin-1])
                    #     print('model: ',model)
                    #     print('eff: ', eff)

        # h1PreselEff.Scale(0.5)  ##rapidity cut correction
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


##------------------ Fill YieldMean histo-----------------------------------------
        if(cclass[1] == 10):
            h1RawCounts.Fit(bw, "I")
            fout = TF1()
            res_hist = YieldMean(h1RawCounts, tmpSyst, fout, bw,
                                 0, 12., 1e-2, 1e-1, False, "log.root", "", "I")
        outDir.cd()
        if(cclass[1] == 10): 
            res_hist.Write()
##--------------------------------------------------------------------------------



##------------------Fill corrected spectrum histo---------------------------------
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
        myCv = TCanvas("ptSpectraCv{}".format(split_string))
        myCv.SetLogy()
        myCv.cd()
        h1RawCounts.Draw()
        tmpSyst.Draw("e2same")
        pinfo2.Draw("x0same")
        myCv.Write()
        myCv.Close()
##----------------------------------------------------------------------------------
        hist_list.append(h1RawCounts.Clone("pT spectrum" + split_string))
        eff_list.append(h1PreselEff.Clone("Efficiecy x Acceptance" + split_string))
        syst_list.append(tmpSyst.Clone("cc"))
        raws_list.append(raws)



##-------------------Spectra in the same canvas ------------------------------------    
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
##-----------------------------------------------------------------------------------


##--------------------Fill summed spectra histo -------------------------------------
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
##-----------------------------------------------------------------------------------



##-------------------Fill efficiencies hist -----------------------------------------
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
##-----------------------------------------------------------------------------------



##-----------------Fill ratio histo -------------------------------------------------
    ratio_error=[]
    for i in range(0,h1RawCounts.GetNbinsX()):
        syst_tot=np.array(raws_list[1][i])/np.array(raws_list[0][i])
        ratio_error.append(float(huber(syst_tot)[1]))
    myCv_ratio = TCanvas("ratio")
    myCv_ratio.cd()
    hist_list[1].Divide(hist_list[0])
    for iBin in range(1, h1RawCounts.GetNbinsX() + 1):
        syst_list[1].SetBinContent(iBin,hist_list[1].GetBinContent(iBin))
        syst_list[1].SetBinError(iBin, ratio_error[iBin-1])    
    hist_list[1].SetTitle(";#it{p}_{T} GeV/#it{c}; {}^{3}_{#bar{#Lambda}} #bar{H} / ^{3}_{#Lambda} H")
    hist_list[1].Draw("P")
    hist_list[1].SetMarkerColor(kBlue)
    hist_list[1].SetLineColor(kBlue)
    syst_list[1].SetLineColor(kBlue)
    hist_list[1].SetMarkerStyle(20)
    syst_list[1].Draw("e2same")
    pinfo2.Draw("x0same")
    myCv_ratio.Write()
    myCv_ratio.Close()
##-----------------------------------------------------------------------------------



resultFile.Close()

bw = -1
pwg = -1
