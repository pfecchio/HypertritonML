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
from scipy import stats

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
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
args = parser.parse_args()

if args.split:
    SPLIT_LIST = ['_matter', '_antimatter']
else:
    SPLIT_LIST = ['']

gROOT.SetBatch()

pol0 = TF1("mypol0", "pol0", 0, 35)

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

var = '#it{ct}'
unit = 'cm'

file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_mass.root'
distribution = TFile(file_name, 'recreate')

file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_mass_res.root'
shift_file = TFile(file_name, 'read')

file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_results_fit.root'
results_file = TFile(file_name, 'read')

hist_shift = shift_file.Get('hist_mean')
shift = []
for iBin in range(1, hist_shift.GetNbinsX() + 1):
    shift.append(hist_shift.GetBinContent(iBin))


mLambda = [1.87561294257,0.00000057]
mDeuton = [1.115683,0.000006]


#for iBin in range(1, hist_shift.GetNbinsX() + 1):
#    shift.append(hist_shift.GetBinContent(iBin))
#if(params["NBODY"]==2):
#    abs_file_name = os.environ['HYPERML_UTILS_{}'.format(params['NBODY'])] + '/he3abs/recCtHe3.root'
#    absorp_file = TFile(abs_file_name)
#    absorp_hist = absorp_file.Get('Reconstructed ct spectrum')

bkgModels = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']
hist_list = []

#loop se mat-anti divise
for split in SPLIT_LIST:
    for cclass in params['CENTRALITY_CLASS']:
        inDirName = f'{cclass[0]}-{cclass[1]}' + split

        h2BDTEff = results_file.Get(f'{inDirName}/BDTeff')
        h1BDTEff = h2BDTEff.ProjectionY("bdteff", 1, 1)

        if(params['NBODY']==2):
            best_sig = np.round(np.array(h1BDTEff)[1:-1], 2)
            sig_ranges = []
            for i in best_sig:
                if i== best_sig[0]:
                    sig_ranges.append([i-0.03, i+0.03, 0.01])
                else:
                    sig_ranges.append([i-0.1, i+0.1, 0.01])
        else:
            best_sig = [0.81, 0.88, 0.83, 0.86, 0.84, 0.85]
            sig_ranges = [[0.70, 90, 0.01], [0.80, 0.95, 0.01], [0.70, 0.90, 0.01], [0.79, 0.94, 0.01], [0.79, 0.90, 0.01], [0.83, 0.90, 0.01]]

        ranges = {
                'BEST': best_sig,
                'SCAN': sig_ranges
        }

        results_file.cd(inDirName)
        out_dir = distribution.mkdir(inDirName)
        cvDir = out_dir.mkdir("canvas")

        hMeanMass = []
        means = []
        errs = []

        h2PreselEff = results_file.Get(f'{inDirName}/PreselEff')
        h1PreselEff = h2PreselEff.ProjectionY("preseleff", 1, 1)

        for model in bkgModels:
            h1MeanMass = h1PreselEff.Clone(f"best_{model}")
            h1MeanMass.Reset()
            if model!="pol2":
                par_index = 3
            else:
                par_index = 4
                
            out_dir.cd()
            for iBin in range(1, h1MeanMass.GetNbinsX() + 1):
                histo = results_file.Get(f'{inDirName}/ct_{params["CT_BINS"][iBin-1]}{params["CT_BINS"][iBin]}/{model}/ct{params["CT_BINS"][iBin-1]}{params["CT_BINS"][iBin]}_pT210_cen090_eff{ranges["BEST"][iBin-1]:.2f}')
                print(f'{inDirName}/ct_{params["CT_BINS"][iBin-1]}{params["CT_BINS"][iBin]}/{model}/ct{params["CT_BINS"][iBin-1]}{params["CT_BINS"][iBin]}_pT210_cen090_eff{ranges["BEST"][iBin-1]:.2f}')

                lineshape = histo.GetFunction("fitTpl")

                h1MeanMass.SetBinContent(iBin,mLambda[0]+mDeuton[0]-(lineshape.GetParameter(par_index)-shift[iBin-1]))
                h1MeanMass.SetBinError(iBin,math.sqrt(lineshape.GetParError(par_index)**2+mLambda[1]**2+mDeuton[1]**2))

                means.append([])
                errs.append([])

                for eff in np.arange(ranges['SCAN'][iBin - 1][0], ranges['SCAN'][iBin - 1][1], ranges['SCAN'][iBin - 1][2]):
                    if eff > 0.99:
                        continue

                    histo = results_file.Get(f'{inDirName}/ct_{params["CT_BINS"][iBin-1]}{params["CT_BINS"][iBin]}/{model}/ct{params["CT_BINS"][iBin-1]}{params["CT_BINS"][iBin]}_pT210_cen090_eff{eff:.2f}')
                    lineshape = histo.GetFunction("fitTpl")

                    means[iBin-1].append(mLambda[0]+mDeuton[0]-(lineshape.GetParameter(par_index)-shift[iBin-1]))
                    errs[iBin-1].append(math.sqrt(lineshape.GetParError(par_index)**2+mLambda[1]**2+mDeuton[1]**2))

            out_dir.cd()
            h1MeanMass.UseCurrentStyle()
            if(split!=""):
                if(model=="pol2"):
                    hist_list.append(h1MeanMass.Clone("hist"+split))
            h1MeanMass.Fit(pol0, "MI0+", "",0,35)
            #fit_function = h1MeanMass.GetFunction("mypol0")
            pol0.SetLineColor(kOrangeC)
            h1MeanMass.Write()
            hMeanMass.append(h1MeanMass)

            cvDir.cd()
            myCv = TCanvas(f"ctSpectraCv_{model}{split}")

            frame = gPad.DrawFrame(
                0., -0.002, 36, 0.001, ";#it{c}t (cm); B_{#Lambda} [GeV/c^{2}]")
            pinfo2 = TPaveText(0.5, 0.65, 0.88, 0.86, "NDC")
            pinfo2.SetBorderSize(0)
            pinfo2.SetFillStyle(0)
            pinfo2.SetTextAlign(22)
            pinfo2.SetTextFont(43)
            pinfo2.SetTextSize(22)
            string1 = '#bf{ALICE Internal}'
            string2 = 'Pb-Pb  #sqrt{#it{s}_{NN}} = 5.02 TeV,  0-90%'
            pinfo2.AddText(string1)
            pinfo2.AddText(string2)
            string = 'B_{#Lambda}'+' = {:.5f} #pm {:.5f} GeV/c^{{2}} '.format(pol0.GetParameter(0), pol0.GetParError(0))
            pinfo2.AddText(string)
            if pol0.GetNDF()is not 0:
                string = f'#chi^{{2}} / NDF = {(pol0.GetChisquare() / pol0.GetNDF()):.2f}'
            pinfo2.AddText(string)
            pol0.Draw("same")
            h1MeanMass.Draw("ex0same")
            #h1MeanMass.SetMarkerStyle(20)
            #h1MeanMass.SetMarkerColor(kBlueC)
            #h1MeanMass.SetLineColor(kBlueC)
            #frame.GetYaxis().SetTitleSize(26)
            #frame.GetYaxis().SetLabelSize(22)
            #frame.GetXaxis().SetTitleSize(26)
            #frame.GetXaxis().SetLabelSize(22)
            #h1MeanMass.SetStats(0)

            pinfo2.Draw("x0same")
            tmpSyst = h1MeanMass.Clone("hSyst")
            corSyst = h1MeanMass.Clone("hCorr")
            tmpSyst.SetFillStyle(0)
            tmpSyst.SetMinimum(0.001)
            tmpSyst.SetMaximum(1000)
            corSyst.SetFillStyle(3345)
            for iBin in range(1, h1MeanMass.GetNbinsX() + 1):
                val = h1MeanMass.GetBinContent(iBin)
                # tmpSyst.SetBinError(iBin, val*0.099)
            #     # corSyst.SetBinError(iBin, 0.086 * val)
            # tmpSyst.SetLineColor(kBlueC)
            # tmpSyst.SetMarkerColor(kBlueC)
            # tmpSyst.Draw("e2same")
            # corSyst.Draw("e2same")
            out_dir.cd()
            myCv.Write()

            #h1MeanMass.Draw("ex0same")
            #pinfo2.Draw()
            cvDir.cd()
            myCv.Write()

        out_dir.cd()

        syst = TH1D("syst", ";B_{#Lambda} (GeV/c^{2});Entries", 300, -0.003, 0.002)
        prob = TH1D("prob", ";constant fit probability;Entries",300, 0, 1)
        tmpCt = hMeanMass[0].Clone("tmpCt")

        combinations = set()
        size = 10000
        count=0
        for _ in range(size):
            tmpCt.Reset()
            comboList = []

            for iBin in range(1, tmpCt.GetNbinsX() + 1):
                index = random.randint(0, len(means[iBin-1])-1)
                comboList.append(index)
                tmpCt.SetBinContent(iBin, means[iBin-1][index])
                tmpCt.SetBinError(iBin, errs[iBin-1][index])

            combo = (x for x in comboList)
            if combo in combinations:
                continue

            combinations.add(combo)
            tmpCt.Fit(pol0, "QMI0+")
            prob.Fill(pol0.GetProb())
            if pol0.GetChisquare() < 3 * pol0.GetNDF():
                if count==0:
                    tmpCt.Write()
                    count=1
                syst.Fill(pol0.GetParameter(0))

        syst.SetFillColor(600)
        syst.SetFillStyle(3345)
        #syst.Scale(1./syst.Integral())
        syst.Write()
        prob.Write()


#-----------------Fill diff histo -------------------------------------------------
if(split!=""):
    myCv_diff = TCanvas("diff")
    myCv_diff.cd()
    hist_list[1].Add(hist_list[0],-1)   
    hist_list[1].Fit("pol0","MI+","",0,35)
    fit_function = hist_list[1].GetFunction("pol0")
    fit_function.SetLineColor(kOrangeC)
    frame = gPad.DrawFrame(
        0., -0.002, 36, 0.001, ";#it{c}t (cm); {}^{3}_{#bar{#Lambda}} #bar{H} - ^{3}_{#Lambda} H")
    pinfo2 = TPaveText(0.5, 0.65, 0.88, 0.86, "NDC")
    pinfo2.SetBorderSize(0)
    pinfo2.SetFillStyle(0)
    pinfo2.SetTextAlign(22)
    pinfo2.SetTextFont(43)
    pinfo2.SetTextSize(22)
    string1 = '#bf{ALICE Internal}'
    string2 = 'Pb-Pb  #sqrt{#it{s}_{NN}} = 5.02 TeV,  0-90%'
    pinfo2.AddText(string1)
    pinfo2.AddText(string2)
    string = '#Deltam'+' = {:.5f} #pm {:.5f} GeV/c^{{2}} '.format(pol0.GetParameter(0), pol0.GetParError(0))
    pinfo2.AddText(string)
    if pol0.GetNDF()is not 0:
        string = f'#chi^{{2}} / NDF = {(pol0.GetChisquare() / pol0.GetNDF()):.2f}'
    pinfo2.AddText(string)
    fit_function.Draw("same")
    hist_list[1].Draw("ex0same")
    pinfo2.Draw("x0same")
    hist_list[1].SetMarkerColor(kBlue)
    hist_list[1].SetLineColor(kBlue)
    hist_list[1].SetMarkerStyle(20)
    myCv_diff.Write()
    myCv_diff.Close()
#-----------------------------------------------------------------------------------

results_file.Close()
