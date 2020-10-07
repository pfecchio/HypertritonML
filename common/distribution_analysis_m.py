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
parser.add_argument('-s', '--scan', help='Use the BDTefficiency selection from the significance scan', action='store_true')
parser.add_argument('-f', '--f', help='Correct the mass using the fit of the resolution', action='store_true')
parser.add_argument('-u', '--unbinned', help='Use the unbinned fits', action='store_true')
args = parser.parse_args()

if args.split:
    SPLIT_LIST = ['_matter','_antimatter']
else:
    SPLIT_LIST = ['']

if args.f:
    SHIFT_NAME = 'opt_fit_shift'
    SHIFT_NAME2D = 'fit_shift'
else:
    SHIFT_NAME = 'opt_shift'
    SHIFT_NAME2D = 'mean_shift'

gROOT.SetBatch()


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

file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_mass_shift.root'
shift_file = TFile(file_name, 'read')

if args.unbinned:
    file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_results_unbinned.root'
    results_file = TFile(file_name, 'read')
    file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_results_fit.root'
    efficiency_file = TFile(file_name, 'read')
else:
    file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_results_fit.root'
    results_file = TFile(file_name, 'read')

file_name = resultsSysDir + '/' + settings['FILE_PREFIX'] + '_results_fit.root'
lambda_file = TFile(file_name, 'read')


MEASUREMENT = ['mass','B_{#Lambda}']
    
mLambda = [1.115683,0.000006]
mDeuton = [1.87561294257,0.00000057]
pol2_meas = []
systerr = []
bkgModels = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']
hist_list = []

#loop se mat-anti divise
for split in SPLIT_LIST:

    if 'ct' in params["FILE_PREFIX"]:
        xlabel = '#it{c}t (cm)'
        if split=="_matter":
            fit_range = [1,35]
        else:
            fit_range = [1,24]
    else:
        xlabel = '#it{p}_{T} (GeV/#it{c})'
        fit_range = [2,9]
    
    pol0 = TF1("mypol0", "pol0", fit_range[0], fit_range[1])

    hist_lambda = lambda_file.Get('0-90'+split+'/mean')

    hist_shift = shift_file.Get(SHIFT_NAME+split)
    print(SHIFT_NAME+split)
    hist_shift2D = shift_file.Get(SHIFT_NAME2D+split)
    shift = []
    for iBin in range(1,10):# hist_shift.GetNbinsX() + 1):
        shift.append(hist_shift.GetBinContent(iBin))
        
    for cclass in params['CENTRALITY_CLASS']:
        inDirName = f'{cclass[0]}-{cclass[1]}' + split
        if args.unbinned:
            h2BDTEff = efficiency_file.Get(f'{inDirName}/BDTeff')
        else:
            h2BDTEff = results_file.Get(f'{inDirName}/BDTeff')
        sig_ranges = []
        if(params['NBODY']==2):
            if args.scan:
                if 'ct' in params["FILE_PREFIX"]:
                    h1BDTEff = h2BDTEff.ProjectionY("bdteff", 1, h2BDTEff.GetNbinsX()+1)
                    best_sig = np.round(np.array(h1BDTEff)[1:-1], 2)
                else:
                    h1BDTEff = h2BDTEff.ProjectionX("bdteff", 1, h2BDTEff.GetNbinsY()+1)
                    best_sig = np.round(np.array(h1BDTEff)[1:-1], 2)
                for i in best_sig:
                    if i== best_sig[0]:
                        sig_ranges.append([i-0.03, i+0.03, 0.01])
                    else:
                        sig_ranges.append([i-0.05, i+0.1, 0.01])
            else:
                best_sig = []
                for i in range(1,len(params['CT_BINS'])*len(params['PT_BINS'])+1):
                    best_sig.append(0.75)
                    sig_ranges.append([0.70, 0.80, 0.01])
            
        else :
            sig_ranges = [[0.70, 90, 0.01], [0.80, 0.95, 0.01], [0.70, 0.90, 0.01], [0.79, 0.94, 0.01], [0.79, 0.90, 0.01], [0.83, 0.90, 0.01]]

        ranges = {
                'BEST': best_sig,
                'SCAN': sig_ranges
        }
        
        if args.unbinned:
            results_file.cd()
        else:
            results_file.cd(inDirName)
        out_dir = distribution.mkdir(inDirName)
        cvDir = out_dir.mkdir("canvas")

        hMeanMass = []
        means = []
        errs = []
        if args.unbinned:
            h2PreselEff = efficiency_file.Get(f'{inDirName}/BDTeff')
        else:    
            h2PreselEff = results_file.Get(f'{inDirName}/BDTeff')
        if 'ct' in params['FILE_PREFIX']:
            h1PreselEff = h2PreselEff.ProjectionY("preseleff", 1, h2PreselEff.GetNbinsX()+1)
        else:
            h1PreselEff = h2PreselEff.ProjectionX("preseleff", 1, h2PreselEff.GetNbinsY()+1)

        for model in bkgModels:
            h1MeanMass = h1PreselEff.Clone(f"best_{model}")
            h1MeanMass.Reset()
            h1Sigmas = h1MeanMass.Clone(f"sigma_{model}{split}")
            h1Sigmas.SetTitle(";#it{p}_{T} (GeV/c);sigma (MeV/c^{2});")
            if model!="pol2":
                par_index = 3
            else:
                par_index = 4
                
            out_dir.cd()
            for iBin in range(1, h1MeanMass.GetNbinsX() + 1):

                if 'ct' in params['FILE_PREFIX']:
                    dir_name = f'{inDirName}/ct_{params["CT_BINS"][iBin-1]}{params["CT_BINS"][iBin]}/'
                    obj_name = f'ct{params["CT_BINS"][iBin-1]}{params["CT_BINS"][iBin]}_pT210_cen090_eff{best_sig[iBin-1]:.2f}'
                else:   
                    dir_name = f'{inDirName}/pt_{params["PT_BINS"][iBin-1]}{params["PT_BINS"][iBin]}/'
                    obj_name = f'ct090_pT{params["PT_BINS"][iBin-1]}{params["PT_BINS"][iBin]}_cen090_eff{best_sig[iBin-1]:.2f}'
                
                if args.unbinned:
                    m_var = results_file.Get('m_'+obj_name+f'_model{model}'+split)
                    sig_var = results_file.Get('sig_'+obj_name+f'_model{model}'+split)
                    mass = m_var.getVal()
                    mass_err = m_var.getError()
                    sigma = sig_var.getVal()
                    sigma_err = sig_var.getError()

                else:
                    histo = results_file.Get(dir_name+f'{model}/'+obj_name)
                    lineshape = histo.GetFunction("fitTpl")
                    mass = lineshape.GetParameter(par_index)
                    mass_err = lineshape.GetParError(par_index)
                    sigma = lineshape.GetParameter(par_index+1)
                    sigma_err = lineshape.GetParError(par_index+1)

                #the shift is in MeV/c^2
                h1MeanMass.SetBinContent(iBin,mass-shift[iBin-1]/1000)
                h1MeanMass.SetBinError(iBin,mass_err)
                h1Sigmas.SetBinContent(iBin,sigma*1000)
                h1Sigmas.SetBinError(iBin,sigma_err*1000)

                means.append([])
                errs.append([])

                for eff in np.arange(ranges['SCAN'][iBin - 1][0], ranges['SCAN'][iBin - 1][1], ranges['SCAN'][iBin - 1][2]):
                    if eff > 0.99:
                        continue

                    if 'ct' in params['FILE_PREFIX']:
                        dir_name = f'{inDirName}/ct_{params["CT_BINS"][iBin-1]}{params["CT_BINS"][iBin]}/'
                        obj_name = f'ct{params["CT_BINS"][iBin-1]}{params["CT_BINS"][iBin]}_pT210_cen090_eff{eff:.2f}'
                    else:   
                        dir_name = f'{inDirName}/pt_{params["PT_BINS"][iBin-1]}{params["PT_BINS"][iBin]}/'
                        obj_name = f'ct090_pT{params["PT_BINS"][iBin-1]}{params["PT_BINS"][iBin]}_cen090_eff{eff:.2f}'
                    
                    if args.unbinned:
                        m_var = results_file.Get('m_'+obj_name+f'_model{model}'+split)
                        print('m_'+obj_name+f'_model{model}'+split)
                        mass = m_var.getVal()
                        mass_err = m_var.getError()
                    else:
                        histo = results_file.Get(dir_name+f'{model}/'+obj_name)
                        lineshape = histo.GetFunction("fitTpl")
                        mass = lineshape.GetParameter(par_index)
                        mass_err = lineshape.GetParError(par_index)
                    means[iBin-1].append(mass-hist_shift2D.GetBinContent(iBin,(int)((eff-0.20)*100+1))/1000)
                    errs[iBin-1].append(mass_err)
            
            for meas in MEASUREMENT:
                out_dir.cd()
                h1MeanMass.UseCurrentStyle()
                if meas == 'B_{#Lambda}':
                    for iBin in range(1, hist_shift.GetNbinsX() + 1):
                        h1MeanMass.SetBinContent(iBin,mLambda[0]+mDeuton[0]-h1MeanMass.GetBinContent(iBin))
                    rangePad = [-0.004,0.002]
                else:
                    rangePad = [2.990,2.995]

                
                if split == '_antimatter':
                    label = {
                    "mass": ['m_{ {}^{3}_{#bar{#Lambda}} #bar{H}}','/c^{2}'],
                    "B_{#Lambda}": ['B_{#bar{#Lambda}}', ''],
                    "title": '{}^{3}_{#bar{#Lambda}} #bar{H}'
                    }
                else:
                    label = {
                    "mass": ['m_{ {}^{3}_{#Lambda}H}','/c^{2}'],
                    "B_{#Lambda}": ['B_{#Lambda}',''],
                    "title": '{}^{3}_{#Lambda} H'
                    }

                if(split!=""):
                    if(model=="pol2"):
                        hist_list.append(h1MeanMass.Clone("hist"+split))
                h1MeanMass.Fit(pol0, "MI0+", "",fit_range[0],fit_range[1])
                #fit_function = h1MeanMass.GetFunction("mypol0")
                pol0.SetLineColor(kOrangeC)
                h1MeanMass.SetMarkerStyle(20)
                h1MeanMass.SetMarkerColor(kBlueC)
                h1MeanMass.SetLineColor(kBlueC)
                h1MeanMass.Write()
                hMeanMass.append(h1MeanMass)

                cvDir.cd()
                myCv = TCanvas(f"ctSpectraCv_{model}_{meas}{split}")
                
                if 'pt' in params['FILE_PREFIX']:
                    xplot_range = [params['PT_BINS'][0],params['PT_BINS'][len(params['PT_BINS'])-1]]
                else:
                    xplot_range = [params['CT_BINS'][0],params['CT_BINS'][len(params['CT_BINS'])-1]]

                frame = gPad.DrawFrame(
                    xplot_range[0], rangePad[0], xplot_range[1], rangePad[1], label['title']+";"+xlabel+";"+ label[meas][0] +"[ GeV"+label[meas][1]+"]")
                pinfo2 = TPaveText(0.5, 0.65, 0.88, 0.86, "NDC")
                pinfo2.SetBorderSize(0)
                pinfo2.SetFillStyle(0)
                pinfo2.SetTextAlign(22)
                pinfo2.SetTextFont(43)
                pinfo2.SetTextSize(33)
                string1 = '#bf{ALICE Internal}'
                string2 = 'Pb-Pb  #sqrt{#it{s}_{NN}} = 5.02 TeV,  0-90%'
                pinfo2.AddText(string1)
                pinfo2.AddText(string2)
                string = label[meas][0] + ' = {:.2f} #pm {:.2f} '.format(pol0.GetParameter(0)*10**(3), pol0.GetParError(0)*10**(3))+'MeV'+label[meas][1]
                if bkgModels=='pol2':
                    pol2_meas.append([pol0.GetParameter(0),pol0.GetParError(0)]) 
                pinfo2.AddText(string)
                if pol0.GetNDF()is not 0:
                    string = f'#chi^{{2}} / NDF = {(pol0.GetChisquare() / pol0.GetNDF()):.2f}'
                pinfo2.AddText(string)
                pol0.Draw("same")
                h1MeanMass.Draw("ex0same")
                h1MeanMass.GetYaxis().SetTitleSize(26)
                h1MeanMass.GetYaxis().SetLabelSize(100)
                h1MeanMass.GetXaxis().SetTitleSize(26)
                h1MeanMass.GetXaxis().SetLabelSize(100)
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
                
                h1Sigmas.SetMarkerStyle(22)
                h1Sigmas.Write()
                #h1MeanMass.Draw("ex0same")
                #pinfo2.Draw()
                cvDir.cd()
                myCv.Write()

        out_dir.cd()

        if split == "_antimatter":
            title = "{}^{3}_{#bar{#Lambda}} #bar{H}"
        else:
            title = "{}^{3}_{#Lambda} H"
        syst = TH1D("syst", title + ";mass (GeV/c^{2});Entries", 200, 2.990, 2.992)
        prob = TH1D("prob", ";constant fit probability;Entries",200, 0, 1)
        tmpCt = hMeanMass[0].Clone("tmpCt")

        combinations = set()
        size = 20000
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
            
            tmpCt.Fit(pol0, "QRMI0+","",fit_range[0],fit_range[1])
            prob.Fill(pol0.GetProb())
            if pol0.GetChisquare() < 3. * pol0.GetNDF():
                if count==0:
                    tmpCt.Write()
                    count=1
                syst.Fill(pol0.GetParameter(0))
        
        systerr.append(syst.GetStdDev())

        syst.SetFillColor(600)
        syst.SetFillStyle(3345)
        #syst.Scale(1./syst.Integral())
        syst.Write()
        prob.Write()


results_file.Close()
