#!/usr/bin/env python3

import argparse
import math
import os
import random

import numpy as np
import yaml
from ROOT import (TF1, TH1D, TH2D, TAxis, TCanvas, TColor, TFile, TFrame,
                  TIter, TKey, TPaveText, gDirectory, gPad, gROOT, gStyle,
                  kBlue, kRed)
from scipy import stats

gROOT.SetBatch()

###############################################################################
# define custom colors
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

random.seed(42)

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('-s', '--scan', help='Use the BDTefficiency selection from the significance scan', action='store_true')
parser.add_argument('-f', '--f', help='Correct the mass using the fit of the resolution', action='store_true')
parser.add_argument('-k', '--k', help='Correct the mass using the kernel density', action='store_true')
parser.add_argument('-u', '--unbinned', help='Use the unbinned fits', action='store_true')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
###############################################################################

###############################################################################
# define some globals
FILE_PREFIX = params['FILE_PREFIX']

MEASUREMENT = ['mass','B_{#Lambda}']
LAMBDA_MASS = [1.115683, 0.000006]
DEUTERON_MASS = [1.87561294257, 0.00000057]

SPLIT_LIST = ['']

if args.split:
    SPLIT_LIST = ['_matter','_antimatter']

SHIFT_NAME = 'opt_shift'
SHIFT_NAME2D = 'mean_shift'

if args.f:
    SHIFT_NAME = 'opt_fit_shift'
    SHIFT_NAME2D = 'fit_shift'

if args.k:
    SHIFT_NAME = 'opt_kernel_shift'


BKG_MODELS = params['BKG_MODELS'] if 'BKG_MODELS' in params else['expo']

CENT_CLASS = params['CENTRALITY_CLASS'][0]
PT_BINS = params['PT_BINS']
###############################################################################

###############################################################################
results_dir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

# output file
file_name = results_dir + '/' + FILE_PREFIX + '_mass.root'
output_file = TFile(file_name, 'recreate')

# mass shift correction file
file_name = results_dir + '/' + FILE_PREFIX + '_mass_shift.root'
mass_correction_file = TFile(file_name, 'read')

# input file depending on the analysis mode
if args.unbinned:
    file_name = results_dir + '/' + FILE_PREFIX + '_results_unbinned.root'

else:
    file_name = results_dir + '/' + FILE_PREFIX + '_results_fit.root'

results_file = TFile(file_name, 'read')

# significance scan output
file_name = results_dir + f'/Efficiencies/{FILE_PREFIX}_sigscan.npy'
sigscan_dict = np.load(file_name, allow_pickle=True).item()
###############################################################################

xlabel = '#it{p}_{T} (GeV/#it{c})'
fit_range = [2, 8]

# loop for split analysis
for split in SPLIT_LIST:
    # # get preselection efficiency file
    # file_name = results_dir + f'/Efficiencies/PreselEff_cent{CENT_CLASS[0]}{CENT_CLASS[1]}{split}.root'
    # presel_eff_file = TFile(file_name, 'read')

    # presel_eff_h2 = presel_eff_file.Get('PreselEff')
    # presel_eff_h1 = presel_eff_h2.ProjectionX('preseleff', 1, presel_eff_h2.GetNbinsY()+1)
    # presel_eff = np.array(presel_eff_h1)[1:-1]

    pol0 = TF1('mypol0', 'pol0', fit_range[0], fit_range[1])

    hist_shift = mass_correction_file.Get(SHIFT_NAME + split)
    hist_shift_h2 = mass_correction_file.Get(SHIFT_NAME2D + split)

    shift = np.array(hist_shift)[1:-1]
        
    input_dirname = f'{CENT_CLASS[0]}-{CENT_CLASS[1]}' + split

    if args.scan:
        best_sig = np.round(list(sigscan_dict.values()), 2)
    else:
        best_sig = np.full(5, 0.60)

    sig_ranges = np.array([best_sig - 0.05, best_sig + 0.05, np.full(len(best_sig), 0.01)])
    ranges = {'BEST': best_sig, 'SCAN': sig_ranges}
        
    if args.unbinned:
        results_file.cd()

    else:
        results_file.cd(input_dirname)

    out_dir = output_file.mkdir(input_dirname)

    hMeanMass = []
    means = []
    errs = []

    for model in BKG_MODELS:
        mass_h1 = hist_shift.Clone(f'mass_best_{model}')
        mass_h1.Reset()

        blambda_h1 = hist_shift.Clone(f'blambda_best_{model}')
        blambda_h1.Reset()

        sigma_h1 = mass_h1.Clone(f'sigma_{model}{split}')
        sigma_h1.SetTitle(";#it{p}_{T} (GeV/c);sigma (MeV/c^{2});")

        par_index = 4 if model is 'pol2' else 3
                
        out_dir.cd()

        for bin_idx in range(1, mass_h1.GetNbinsX() + 1):
            if args.unbinned:
                sub_dir_name = f'pT{PT_BINS[bin_idx-1]}{PT_BINS[bin_idx]}_eff{best_sig[bin_idx-1]:.2f}{split}'
                roo_hyp_mass = results_file.Get(sub_dir_name + f'/hyp_mass_model{model}')
                roo_width = results_file.Get(sub_dir_name + f'/width_model{model}')
                hyp_mass = roo_hyp_mass.getVal()
                hyp_mass_err = roo_hyp_mass.getError()
                width = roo_width.getVal()
                width_err = roo_width.getError()

            # else:
            #     histo = results_file.Get(dir_name + f'{model}/' + obj_name)
            #     lineshape = histo.GetFunction("fitTpl")
            #     mass = lineshape.GetParameter(par_index)
            #     mass_err = lineshape.GetParError(par_index)
            #     sigma = lineshape.GetParameter(par_index+1)
            #     sigma_err = lineshape.GetParError(par_index+1)

            mass_h1.SetBinContent(bin_idx, hyp_mass - shift[bin_idx-1]/1000) # mass correction in MeV/c^2
            mass_h1.SetBinError(bin_idx, hyp_mass_err)
            sigma_h1.SetBinContent(bin_idx, width*1000) # width in MeV/c^2
            sigma_h1.SetBinError(bin_idx, width_err*1000)  # width in MeV/c^2
            blambda_h1.SetBinContent(bin_idx, (LAMBDA_MASS[0] + DEUTERON_MASS[0] - mass_h1.GetBinContent(bin_idx))*1000)  # BLambda in MeV
            blambda_error = math.sqrt(LAMBDA_MASS[1]**2 + DEUTERON_MASS[1]**2 + mass_h1.GetBinError(bin_idx)**2)
            blambda_h1.SetBinError(bin_idx, mass_h1.GetBinError(bin_idx)*1000)
            

            # means.append([])
            # errs.append([])

            # # sampling for systematics ?
            # for eff in np.arange(ranges['SCAN'][bin_idx - 1][0], ranges['SCAN'][bin_idx - 1][1], ranges['SCAN'][bin_idx - 1][2]):
            #     if eff > 0.99:
            #         continue

            #         if 'ct' in FILE_PREFIX:
            #             dir_name = f'{input_dirname}/ct_{params['CT_BINS'][bin_idx-1]}{params["CT_BINS"][bin_idx]}/'
            #             obj_name = f'ct{params["CT_BINS"][bin_idx-1]}{params["CT_BINS"][bin_idx]}_pT210_cen090_eff{eff:.2f}'

            #         else:   
            #             dir_name = f'{input_dirname}/pt_{PT_BINS[bin_idx-1]}{PT_BINS[bin_idx]}/'
            #             obj_name = f'ct090_pT{PT_BINS[bin_idx-1]}{PT_BINS[bin_idx]}_cen090_eff{eff:.2f}'
                    
            #         if args.unbinned:
            #             m_var = results_file.Get('m_'+obj_name+f'_model{model}'+split)
            #             mass = m_var.getVal()
            #             mass_err = m_var.getError()

            #         else:
            #             histo = results_file.Get(dir_name+f'{model}/'+obj_name)
            #             lineshape = histo.GetFunction("fitTpl")
            #             mass = lineshape.GetParameter(par_index)
            #             mass_err = lineshape.GetParError(par_index)

            #         means[bin_idx-1].append(mass-hist_shift2D.GetBinContent(bin_idx,(int)((eff-0.20)*100+1))/1000)
            #         errs[bin_idx-1].append(mass_err)


        ###############################################################################
        # mass plot
        out_dir.cd()
        mass_h1.UseCurrentStyle()

        pad_range = [2.988, 2.993]
        
        if split is '_antimatter':
            label = 'm_{ {}^{3}_{#bar{#Lambda}} #bar{H}}'
        else:
            label = 'm_{ {}^{3}_{#Lambda}H}'
        mass_h1.Fit(pol0, 'LI0+', '', fit_range[0], fit_range[1])
        pol0.SetLineColor(kOrangeC)
        mass_h1.SetMarkerStyle(20)
        mass_h1.SetMarkerColor(kBlueC)
        mass_h1.SetLineColor(kBlueC)
        mass_h1.Write()
        hMeanMass.append(mass_h1)
        canvas = TCanvas(f"Hyp_Mass_{model}_{split}")
            
        xplot_range = [PT_BINS[0], PT_BINS[-1]]
        frame = gPad.DrawFrame(xplot_range[0], pad_range[0], xplot_range[1], pad_range[1], ';'+xlabel+';'+label+' [ GeV/#it{c}^{2} ]')
        pinfo = TPaveText(0.142, 0.620, 0.522, 0.848, "NDC")
        pinfo.SetBorderSize(0)
        pinfo.SetFillStyle(0)
        pinfo.SetTextAlign(11)
        pinfo.SetTextFont(43)
        pinfo.SetTextSize(25)
        string_list = []
        string_list.append('#bf{ALICE Internal}')
        string_list.append('Pb-Pb  #sqrt{#it{s}_{NN}} = 5.02 TeV,  0-90%')
        string_list.append(label + ' = {:.5f} #pm {:.5f} '.format(pol0.GetParameter(0), pol0.GetParError(0)) + 'GeV/#it{c}^{2}')
        if pol0.GetNDF() is not 0:
            string_list.append(f'#chi^{{2}} / NDF = {(pol0.GetChisquare() / pol0.GetNDF()):.2f}')
        for s in string_list:
            pinfo.AddText(s)

        pol0.Draw("same")
        pinfo.Draw("x0same")
        mass_h1.Draw("ex0same")
        mass_h1.GetYaxis().SetTitleSize(26)
        mass_h1.GetYaxis().SetLabelSize(100)
        mass_h1.GetXaxis().SetTitleSize(26)
        mass_h1.GetXaxis().SetLabelSize(100)
        canvas.Write()
                
        sigma_h1.SetMarkerStyle(22)
        sigma_h1.SetMarkerColor(kBlue)
        sigma_h1.Write()
        mass_h1.Write()

        ###############################################################################
        # B_Lambda plot
        pad_range = [-2.05, 2.05]
            
        if split is '_antimatter':
            label = 'B_{#bar{#Lambda}}'
        else:
            label = 'B_{#Lambda}'

        blambda_h1.Fit(pol0, 'LI0+', '', fit_range[0], fit_range[1])

        pol0.SetLineColor(kOrangeC)
        blambda_h1.SetMarkerStyle(20)
        blambda_h1.SetMarkerColor(kBlueC)
        blambda_h1.SetLineColor(kBlueC)
        blambda_h1.Write()
        hMeanMass.append(blambda_h1)

        canvas = TCanvas(f"BLambda_{model}_{split}")
                
        xplot_range = [PT_BINS[0], PT_BINS[-1]]

        frame = gPad.DrawFrame(xplot_range[0], pad_range[0], xplot_range[1], pad_range[1], ';'+xlabel+';'+label+' [ MeV/#it{c}^{2} ]')
        pinfo = TPaveText(0.142, 0.620, 0.522, 0.848, "NDC")
        pinfo.SetBorderSize(0)
        pinfo.SetFillStyle(0)
        pinfo.SetTextAlign(11)
        pinfo.SetTextFont(43)
        pinfo.SetTextSize(25)

        string_list = []
        string_list.append('#bf{ALICE Internal}')
        string_list.append('Pb-Pb  #sqrt{#it{s}_{NN}} = 5.02 TeV,  0-90%')
        string_list.append(label + ' = {:.2f} #pm {:.2f} '.format(pol0.GetParameter(0), pol0.GetParError(0)) + 'MeV/#it{c}^{2}')

        if pol0.GetNDF() is not 0:
            string_list.append(f'#chi^{{2}} / NDF = {(pol0.GetChisquare() / pol0.GetNDF()):.2f}')

        for s in string_list:
            pinfo.AddText(s)

        pol0.Draw("same")
        pinfo.Draw("x0same")
        blambda_h1.Draw("ex0same")
        blambda_h1.GetYaxis().SetTitleSize(26)
        blambda_h1.GetYaxis().SetLabelSize(100)
        blambda_h1.GetXaxis().SetTitleSize(26)
        blambda_h1.GetXaxis().SetLabelSize(100)

        canvas.Write()
        blambda_h1.Write()

        # if split == "_antimatter":
        #     title = "{}^{3}_{#bar{#Lambda}} #bar{H}"

        # else:
        #     title = "{}^{3}_{#Lambda} H"

        # syst = TH1D("syst", title + ";mass (GeV/c^{2});Entries", 200, 2.990, 2.992)
        # prob = TH1D("prob", ";constant fit probability;Entries",200, 0, 1)
        # tmpCt = hMeanMass[0].Clone("tmpCt")

        # combinations = set()
        # size = 20000
        # count = 0
        
        # for _ in range(size):
        #     tmpCt.Reset()
        #     comboList = []

        #     for bin_idx in range(1, tmpCt.GetNbinsX() + 1):
        #         index = random.randint(0, len(means[bin_idx-1])-1)
        #         comboList.append(index)
        #         tmpCt.SetBinContent(bin_idx, means[bin_idx-1][index])
        #         tmpCt.SetBinError(bin_idx, errs[bin_idx-1][index])

        #     combo = (x for x in comboList)

        #     if combo in combinations:
        #         continue

        #     combinations.add(combo)
            
        #     tmpCt.Fit(pol0, "QRMI0+","",fit_range[0],fit_range[1])
        #     prob.Fill(pol0.GetProb())

        #     if pol0.GetChisquare() < 3. * pol0.GetNDF():
        #         if count==0:
        #             tmpCt.Write()
        #             count = 1
                    
        #         syst.Fill(pol0.GetParameter(0))
        
        # systerr.append(syst.GetStdDev())

        # syst.SetFillColor(600)
        # syst.SetFillStyle(3345)
        # #syst.Scale(1./syst.Integral())
        # syst.Write()
        # prob.Write()

results_file.Close()
