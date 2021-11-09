#!/usr/bin/env python3

import argparse
import math
import os

import hyp_analysis_utils as hau
import hyp_plot_utils as hpu
import numpy as np
import pandas as pd
import yaml
from scipy import stats

import ROOT
ROOT.gROOT.SetBatch()
np.random.seed(42)
ROOT.gROOT.LoadMacro('RooCustomPdfs/RooDSCBShape.cxx++')
from ROOT import RooDSCBShape


###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
parser.add_argument('-s', '--significance', help='Use the BDTefficiency selection from the significance scan', action='store_true')
parser.add_argument('-syst', '--systematics', help='Run systematic uncertanties estimation', action='store_true')
parser.add_argument('-dbshape', '--dbshape', help='Fit using DSCBShape', action='store_true')
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

DATA_PATH = os.path.expandvars(params['DATA_PATH'])

BKG_MODELS = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']
CENT_CLASS = params['CENTRALITY_CLASS'][0]
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
EFF_ARRAY = np.around(np.arange(EFF_MIN, EFF_MAX+EFF_STEP, EFF_STEP), 2)

SIGNIFICANCE_SCAN = args.significance
SYSTEMATICS = args.systematics
DBSHAPE = args.dbshape

SYSTEMATICS_COUNTS = 10000
FIX_EFF = 0.70 if not SIGNIFICANCE_SCAN else 0
###############################################################################

###############################################################################
# input/output files
results_dir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]
tables_dir = os.path.dirname(DATA_PATH)
efficiency_dir = os.environ['HYPERML_EFFICIENCIES_{}'.format(params['NBODY'])]

# significance scan output
file_name = results_dir + f'/Efficiencies/{FILE_PREFIX}_sigscan.npy'
sigscan_dict = np.load(file_name, allow_pickle=True).item()


suffix = "" if not DBSHAPE else "_dscb"
# output file
file_name = results_dir + f'/{FILE_PREFIX}_lifetime{suffix}.root'
output_file = ROOT.TFile(file_name, 'recreate')
###############################################################################

file_name = results_dir + f'/{FILE_PREFIX}_signal_extraction{suffix}.root'
input_file = ROOT.TFile(file_name)

###############################################################################

file_name = efficiency_dir + f'/{FILE_PREFIX}_preseleff_cent090.root'
efficiency_file = ROOT.TFile(file_name, 'read')
EFFICIENCY = efficiency_file.Get('PreselEff').ProjectionY()
EFFICIENCY.SetDirectory(0)


file_name = results_dir + '/He3_abs_1.5.root'
abs_file = ROOT.TFile(file_name, 'read')
ABSORPTION = abs_file.Get('0_90/fEffCt_antimatter_cent_0_90_func_BGBW')
ABSORPTION.SetDirectory(0)


file_name = results_dir + '/He3_abs_test.root'
abs_file = ROOT.TFile(file_name, 'read')
MATTER_ABSORPTION = abs_file.Get('0_90/matter_antimatter_ratio')
MATTER_ABSORPTION.SetDirectory(0)





###############################################################################
# define support globals and methods for getting hypertriton counts
RAW_COUNTS_H2 = {}
RAW_COUNTS_BEST = {}

CORRECTED_COUNTS_H2 = {}
CORRECTED_COUNTS_BEST = {}



# prepare histograms for the analysis
for model in BKG_MODELS:
        RAW_COUNTS_H2[model] = input_file.Get(f'raw_counts_{model}')
        RAW_COUNTS_BEST[model] = RAW_COUNTS_H2[model].ProjectionX(f'raw_counts_best_{model}')
        CORRECTED_COUNTS_H2[model] = RAW_COUNTS_H2[model].Clone('corrected_counts')
        CORRECTED_COUNTS_BEST[model] =  RAW_COUNTS_BEST[model].Clone('corrected_counts_best')


def get_presel_eff(ctbin):
    return EFFICIENCY.GetBinContent(EFFICIENCY.FindBin((ctbin[0] + ctbin[1]) / 2))


def get_absorption_correction(ctbin):
    abso = ABSORPTION.GetBinContent(ABSORPTION.FindBin((ctbin[0] + ctbin[1]) / 2))
    matter_abso = abso*MATTER_ABSORPTION.GetBinContent(MATTER_ABSORPTION.FindBin((ctbin[0] + ctbin[1]) / 2))
    abso = (abso + matter_abso)/2
    return abso 


def fill_raw(bkg, ctbin, counts, counts_err, eff):
    bin_idx = RAW_COUNTS_H2[bkg].FindBin((ctbin[0] + ctbin[1]) / 2, round(eff + 0.005, 3))
    RAW_COUNTS_H2[bkg].SetBinContent(bin_idx, counts)
    RAW_COUNTS_H2[bkg].SetBinError(bin_idx, counts_err)


def fill_raw_best(bkg, ctbin, counts, counts_err, eff):
    bin_idx = RAW_COUNTS_BEST[bkg].FindBin((ctbin[0] + ctbin[1]) / 2)
    RAW_COUNTS_BEST[bkg].SetBinContent(bin_idx, counts)
    RAW_COUNTS_BEST[bkg].SetBinError(bin_idx, counts_err)


def fill_corrected(bkg, ctbin, counts, counts_err, eff):
    bin_idx = CORRECTED_COUNTS_H2[bkg].FindBin((ctbin[0] + ctbin[1]) / 2, round(eff + 0.005, 3))
    bin_idx1d = CORRECTED_COUNTS_BEST[bkg].FindBin((ctbin[0] + ctbin[1]) / 2)

    abs_corr = get_absorption_correction(ctbin)
    presel_eff = get_presel_eff(ctbin)
    bin_width = CORRECTED_COUNTS_BEST[bkg].GetBinWidth(bin_idx1d)

    CORRECTED_COUNTS_H2[bkg].SetBinContent(bin_idx, counts/eff/presel_eff/abs_corr/bin_width)
    CORRECTED_COUNTS_H2[bkg].SetBinError(bin_idx, counts_err/eff/presel_eff/abs_corr/bin_width)
    

def fill_corrected_best(bkg, ctbin, counts, counts_err, eff):
    bin_idx = CORRECTED_COUNTS_BEST[bkg].FindBin((ctbin[0] + ctbin[1]) / 2)

    abs_corr = get_absorption_correction(ctbin)
    presel_eff = get_presel_eff(ctbin)
    bin_width = CORRECTED_COUNTS_BEST[bkg].GetBinWidth(bin_idx)

    CORRECTED_COUNTS_BEST[bkg].SetBinContent(bin_idx, counts/eff/presel_eff/abs_corr/bin_width)
    CORRECTED_COUNTS_BEST[bkg].SetBinError(bin_idx, counts_err/eff/presel_eff/abs_corr/bin_width)


def get_signscan_eff(ctbin):
    key = f'ct{ctbin[0]}{ctbin[1]}pt{PT_BINS[0]}{PT_BINS[1]}'
    return sigscan_dict[key]
    

def get_eff_index(eff):
    idx = (eff - EFF_MIN + EFF_STEP) * 100
    if isinstance(eff, np.ndarray):
        return idx.astype(int)

    return int(idx)


def get_corrected_counts(bkg, ctbin, eff):
    bin_idx = CORRECTED_COUNTS_H2[bkg].FindBin((ctbin[0] + ctbin[1]) / 2, round(eff + 0.005, 3))

    counts = CORRECTED_COUNTS_H2[bkg].GetBinContent(bin_idx)
    error = CORRECTED_COUNTS_H2[bkg].GetBinError(bin_idx)
    
    return counts, error

def get_measured_h2(h2, bkg, ctbin, eff):
    bin_idx = h2[bkg].FindBin((ctbin[0] + ctbin[1]) / 2, round(eff + 0.005, 3))
    var = h2[bkg].GetBinContent(bin_idx)
    error = h2[bkg].GetBinError(bin_idx)
    return var, error



def get_effscore_dict(ctbin):
    info_string = f'090_210_{ctbin[0]}{ctbin[1]}'
    file_name = efficiency_dir + f'/Eff_Score_{info_string}.npy'
    return {round(e[0], 2): e[1] for e in np.load(file_name).T}


###############################################################################
# significance-scan/fixed efficiencies switch

if not SIGNIFICANCE_SCAN:
    eff_best_array = np.full(len(CT_BINS) - 1, FIX_EFF)
else:
    eff_best_array = [round(sigscan_dict[f'ct{ctbin[0]}{ctbin[1]}pt210'][0], 2) for ctbin in zip(CT_BINS[:-1], CT_BINS[1:])]

# efficiency ranges for sampling the systematics
syst_eff_ranges = np.asarray([list(range(int(x * 100) - 10, int(x * 100) + 11)) for x in eff_best_array]) / 100
# define the expo function for the lifetime fit
expo = ROOT.TF1('myexpo', '[0]*exp(-x/([1]*0.029979245800))/([1]*0.029979245800)', 0, 35)
expo.SetParLimits(1, 100, 5000)
#################################################


for index, ctbin in enumerate(zip(CT_BINS[:-1], CT_BINS[1:])):
    bdt_eff_best = round(sigscan_dict[f'ct{ctbin[0]}{ctbin[1]}pt210'][0], 2)
    presel_eff = get_presel_eff(ctbin)
    print("BDT eff best: ", bdt_eff_best)
    print("Presel eff: ", presel_eff)
    for bdt_eff in syst_eff_ranges[index]:
        for model in BKG_MODELS:
            raw_counts, raw_counts_error = get_measured_h2(RAW_COUNTS_H2, model, ctbin, bdt_eff)
            print("Raw counts: ", raw_counts)
            fill_corrected(model, ctbin, raw_counts, raw_counts_error, bdt_eff)
            if bdt_eff == bdt_eff_best:
                fill_corrected_best(model, ctbin, raw_counts, raw_counts_error, bdt_eff)



if SYSTEMATICS:
    # systematics histos
    lifetime_dist = ROOT.TH1D('syst_lifetime', ';#tau ps ;counts', 100, 150, 350)
    lifetime_prob = ROOT.TH1D('prob_lifetime', ';prob. ;counts', 100, 0, 1)

    tmp_ctdist = CORRECTED_COUNTS_BEST[BKG_MODELS[0]].Clone('tmp_ctdist')

    combinations = set()
    sample_counts = 0   # good fits
    iterations = 0  # total fits

    # stop with SYSTEMATICS_COUNTS number of good B_{Lambda} fits
    while sample_counts < SYSTEMATICS_COUNTS:
        tmp_ctdist.Reset()

        iterations += 1

        bkg_list = []
        eff_list = []
        bkg_idx_list = []
        eff_idx_list = []

        # loop over ctbins
        for ctbin_idx in range(len(CT_BINS) - 1):
            # random bkg model
            bkg_index = np.random.randint(0, len(BKG_MODELS))
            bkg_idx_list.append(bkg_index)
            bkg_list.append(BKG_MODELS[bkg_index])

            # random BDT efficiency in the defined range
            eff = np.random.choice(syst_eff_ranges[ctbin_idx])
            eff_list.append(eff)
            eff_idx = get_eff_index(eff)
            eff_idx_list.append(eff_idx)

        # convert indexes into hash and if already sampled skip this combination
        combo = ''.join(map(str, bkg_idx_list + eff_idx_list))
        if combo in combinations:
            continue

        # if indexes are good measure lifetime
        ctbin_idx = 1
        ct_bin_it = iter(zip(CT_BINS[:-1], CT_BINS[1:]))

        for model, eff in zip(bkg_list, eff_list):
            ctbin = next(ct_bin_it)

            counts, error = get_corrected_counts(model, ctbin, eff)

            tmp_ctdist.SetBinContent(ctbin_idx, counts)
            tmp_ctdist.SetBinError(ctbin_idx, error)

            ctbin_idx += 1

        tmp_ctdist.Fit(expo, 'MEI0+', '', 0, 35)

        # if ct fit is good use it for systematics
        if expo.GetChisquare() > 2. * expo.GetNDF():
            continue

        lifetime_dist.Fill(expo.GetParameter(1))
        lifetime_prob.Fill(expo.GetProb())

        combinations.add(combo)
        sample_counts += 1

    output_file.cd()

    lifetime_dist.Write()
    lifetime_prob.Write()

    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'\nGood iterations / Total iterations -> {SYSTEMATICS_COUNTS/iterations:.4f}')
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')



kBlueC = ROOT.TColor.GetColor('#1f78b4')
kBlueCT = ROOT.TColor.GetColorTransparent(kBlueC, 0.5)
kRedC = ROOT.TColor.GetColor('#e31a1c')
kRedCT = ROOT.TColor.GetColorTransparent(kRedC, 0.5)



for model in BKG_MODELS:
    output_file.cd()

    RAW_COUNTS_H2[model].Write()
    RAW_COUNTS_BEST[model].Write()

    CORRECTED_COUNTS_H2[model].Write()
    CORRECTED_COUNTS_BEST[model].Write()

    CORRECTED_COUNTS_BEST[model].UseCurrentStyle()
    CORRECTED_COUNTS_BEST[model].Fit(expo, 'MEI0+', '', 0, 35)

    fit_function = CORRECTED_COUNTS_BEST[model].GetFunction('myexpo')
    fit_function.SetLineColor(kRedC)

    canvas = ROOT.TCanvas(f'ct_spectra_{model}')
    canvas.SetLogy()

    frame = ROOT.gPad.DrawFrame(-0.5, 1, 35.5, 2000, ';#it{c}t (cm);d#it{N}/d(#it{c}t) [(cm)^{-1}]')

    pinfo = ROOT.TPaveText(0.5, 0.65, 0.88, 0.86, 'NDC')
    pinfo.SetBorderSize(0)
    pinfo.SetFillStyle(0)
    pinfo.SetTextAlign(22)
    pinfo.SetTextFont(43)
    pinfo.SetTextSize(22)

    strings = []
    strings.append('#bf{ALICE Internal}')
    strings.append('Pb-Pb  #sqrt{#it{s}_{NN}} = 5.02 TeV,  0-90%')
    strings.append(f'#tau = {fit_function.GetParameter(1):.0f} #pm {fit_function.GetParError(1):.0f} ps')
    strings.append(f'#chi^{{2}} / NDF = {(fit_function.GetChisquare() / fit_function.GetNDF()):.2f}')

    for s in strings:
        pinfo.AddText(s)

    fit_function.Draw('same')
    CORRECTED_COUNTS_BEST[model].Draw('ex0same')
    CORRECTED_COUNTS_BEST[model].SetMarkerStyle(20)
    CORRECTED_COUNTS_BEST[model].SetMarkerColor(kBlueC)
    CORRECTED_COUNTS_BEST[model].SetLineColor(kBlueC)
    CORRECTED_COUNTS_BEST[model].SetMinimum(0.001)
    CORRECTED_COUNTS_BEST[model].SetMaximum(1000)
    CORRECTED_COUNTS_BEST[model].SetStats(0)

    frame.GetYaxis().SetTitleSize(26)
    frame.GetYaxis().SetLabelSize(22)
    frame.GetXaxis().SetTitleSize(26)
    frame.GetXaxis().SetLabelSize(22)
    frame.GetYaxis().SetRangeUser(7, 5000)
    frame.GetXaxis().SetRangeUser(0.5, 35.5)

    pinfo.Draw('x0same')

    canvas.Write()

