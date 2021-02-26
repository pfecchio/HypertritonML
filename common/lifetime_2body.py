#!/usr/bin/env python3

import argparse
import math
import os

import hyp_analysis_utils as hau
import hyp_plot_utils as hpu
import numpy as np
import pandas as pd
import ROOT
import yaml
from scipy import stats

ROOT.gROOT.SetBatch()

np.random.seed(42)

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('-s', '--significance', help='Use the BDTefficiency selection from the significance scan', action='store_true')
parser.add_argument('-syst', '--systematics', help='Run systematic uncertanties estimation', action='store_true')
parser.add_argument('-k', '--skipfits', help='Run systematic uncertanties estimation skipping invariant mass fits', action='store_true')
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

SPLIT_LIST = ['_matter','_antimatter'] if args.split else ['']
BKG_MODELS = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']

CENT_CLASS = params['CENTRALITY_CLASS'][0]
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
EFF_ARRAY = np.around(np.arange(EFF_MIN, EFF_MAX, EFF_STEP), 2)

SPLIT_MODE = args.split
SIGNIFICANCE_SCAN = args.significance

SKIP_FITS = args.skipfits
SYSTEMATICS = True if SKIP_FITS else args.systematics

FIX_EFF = 0.70 if not SIGNIFICANCE_SCAN else 0

SYSTEMATICS_COUNTS = 100000
###############################################################################

###############################################################################
# input/output files
results_dir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]
tables_dir = os.path.dirname(DATA_PATH)
efficiency_dir = os.environ['HYPERML_EFFICIENCIES_{}'.format(params['NBODY'])]
utils_dir = os.environ['HYPERML_UTILS_2']

# input data file
file_name = tables_dir + f'/applied_df_{FILE_PREFIX}.parquet.gzip'
data_df = pd.read_parquet(file_name, engine='fastparquet')

# output file
file_name = results_dir + f'/{FILE_PREFIX}.root'
output_file = ROOT.TFile(file_name, 'update' if SKIP_FITS else 'recreate')

# mc file
file_name = tables_dir + f'/applied_mc_df_{FILE_PREFIX}.parquet.gzip'
mc_df = pd.read_parquet(file_name, engine='fastparquet')

# preselection eff
file_name = efficiency_dir + f'/{FILE_PREFIX}_preseleff_cent090.root'
efficiency_file = ROOT.TFile(file_name, 'read')
EFFICIENCY = efficiency_file.Get('PreselEff').ProjectionY()

# absorption correction file
file_name = utils_dir + '/he3abs/recCtHe3.root'
abs_file = ROOT.TFile(file_name, 'read')
ABSORPTION = abs_file.Get('Reconstructed ct spectrum')

# significance scan output
file_name = results_dir + f'/Efficiencies/{FILE_PREFIX}_sigscan.npy'
sigscan_dict = np.load(file_name, allow_pickle=True).item()
###############################################################################

###############################################################################
# define support globals and methods for getting hypertriton counts
RAW_COUNTS_H2 = {}
RAW_COUNTS_BEST = {}

CORRECTED_COUNTS_H2 = {}
CORRECTED_COUNTS_BEST = {}

MC_MASS = 2.99131

KDE_SAMPLE_SIZE = 20000
BANDWIDTH = 1.905

for split in SPLIT_LIST:
    for model in BKG_MODELS:
        # initialize histos for the hypertriton counts vs ct vs BDT efficiency
        if SKIP_FITS:
            CORRECTED_COUNTS_H2[model] = output_file.Get(f'corrected_counts_{model}{split}')
            CORRECTED_COUNTS_BEST[model] = output_file.Get(f'corrected_counts_best_{model}{split}')

        else:
            RAW_COUNTS_H2[model] = ROOT.TH2D(f'raw_counts_{model}', '', len(CT_BINS)-1, np.array(CT_BINS, 'double'), len(EFF_ARRAY)-1, np.array(EFF_ARRAY, 'double'))
            RAW_COUNTS_BEST[model] = RAW_COUNTS_H2[model].ProjectionX(f'raw_counts_best_{model}{split}')

            CORRECTED_COUNTS_H2[model] = RAW_COUNTS_H2[model].Clone(f'corrected_counts_{model}{split}')
            CORRECTED_COUNTS_BEST[model] = RAW_COUNTS_BEST[model].Clone(f'corrected_counts_best_{model}{split}')


def get_presel_eff(ctbin):
    return EFFICIENCY.GetBinContent(EFFICIENCY.FindBin((ctbin[0] + ctbin[1]) / 2))


def get_absorption_correction(ctbin):
    bin_idx = ABSORPTION.FindBin((ctbin[0] + ctbin[1]) / 2)
    return 1 - ABSORPTION.GetBinContent(bin_idx)


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

eff_best_it = iter(eff_best_array)
eff_range_it = iter(syst_eff_ranges)

# define the expo function for the lifetime fit
expo = ROOT.TF1('myexpo', '[0]*exp(-x/([1]*0.029979245800))/([1]*0.029979245800)', 0, 35)
expo.SetParLimits(1, 100, 5000)

# actual analysis
for split in SPLIT_LIST:
    if SKIP_FITS:
        continue
    for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
        score_dict = get_effscore_dict(ctbin)

        # get data slice for this ct bin
        data_slice = data_df.query('@ctbin[0]<ct<@ctbin[1]')
        mc_slice = mc_df.query('@ctbin[0]<ct<@ctbin[1]')

        subdir_name = f'ct{ctbin[0]}{ctbin[1]}'
        ct_dir = output_file.mkdir(subdir_name)
        ct_dir.cd()

        eff_best = next(eff_best_it)
        eff_range = next(eff_range_it)

        print(eff_range)
        print()
        for eff in eff_range:
            # define global RooFit objects
            mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2.960, 3.040, 'GeV/c^{2}')
            delta_mass = ROOT.RooRealVar('delta_m', '#Delta m', -0.005, 0.005, 'GeV/c^{2}')
            shift_mass = ROOT.RooAddition('shift_m', "m + #Delta m", ROOT.RooArgList(mass, delta_mass))

            mass_set = ROOT.RooArgSet(mass)
            mass_norm_set = ROOT.RooFit.NormSet(mass_set)
               
            # get the data slice as a RooDataSet
            tsd = score_dict[eff]
            print(eff)
            roo_data_slice = hau.ndarray2roo(np.array(data_slice.query('score>@tsd')['m'].values, dtype=np.float64), mass)

            mc_array = np.array(mc_slice.query('score>@tsd')['m'].values, dtype=np.float64)
            np.random.shuffle(mc_array)
            roo_mc_slice = hau.ndarray2roo(mc_array[:KDE_SAMPLE_SIZE], mass)

            for model in BKG_MODELS:
                # define signal parameters
                mu = ROOT.RooRealVar('mu', 'hypertriton mass', 2.989, 2.993, 'GeV/c^{2}')
                sigma = ROOT.RooRealVar('sigma', 'hypertriton width', 0.0001, 0.004, 'GeV/c^{2}')

                # define temporary signal component for signal region determination
                signal_tmp = ROOT.RooGaussian('signal', 'signal component pdf', mass, mu, sigma)

                # define background parameters
                slope = ROOT.RooRealVar('slope', 'exponential slope', -100., 100.)

                c0 = ROOT.RooRealVar('c0', 'constant c0', -1., 1.)
                c1 = ROOT.RooRealVar('c1', 'constant c1', -1., 1.)

                # define background component depending on background model required
                if model == 'pol1':
                    background = ROOT.RooPolynomial('bkg', 'pol1 bkg', mass, ROOT.RooArgList(c0))

                if model == 'pol2':
                    background = ROOT.RooPolynomial('bkg', 'pol2 bkg', mass, ROOT.RooArgList(c0, c1))

                if model == 'expo':
                    background = ROOT.RooExponential('bkg', 'expo bkg', mass, slope)

                # define fraction
                n_sig = ROOT.RooRealVar('n_sig', 'number of signal candidates', 0, 2000)
                n_bkg = ROOT.RooRealVar('n_bkg', 'number of background candidates', 0, 2000)

                # define the fit funciton -> temorary signal component + background component
                fit_function_tmp = ROOT.RooAddPdf(f'{model}_gaus', 'signal + background', ROOT.RooArgList(signal_tmp, background), ROOT.RooArgList(n_sig, n_bkg))

                # fit data
                fit_results_tmp = fit_function_tmp.fitTo(roo_data_slice, ROOT.RooFit.Range(2.960, 3.040), ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.NumCPU(32), ROOT.RooFit.Save())

                mu.setConstant(ROOT.kTRUE)
                mu.removeError()
                sigma.setConstant(ROOT.kTRUE)
                sigma.removeError()
                
                # define the signal component
                mass.setVal(MC_MASS)
                signal = ROOT.RooKeysPdf('signal', 'signal', shift_mass, mass, roo_mc_slice, ROOT.RooKeysPdf.NoMirror, BANDWIDTH)

                # define the fit funciton -> signal component + background component
                fit_function = ROOT.RooAddPdf(f'{model}_gaus', 'signal + background', ROOT.RooArgList(signal, background), ROOT.RooArgList(n_sig, n_bkg))

                fit_results = fit_function.fitTo(roo_data_slice, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.Range(2.960, 3.040), ROOT.RooFit.NumCPU(32), ROOT.RooFit.Save())

                frame = mass.frame(60)
                frame.SetName(f'eff{eff:.2f}_{model}')

                roo_data_slice.plotOn(frame, ROOT.RooFit.Name('data'))
                fit_function.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Name('model'))
                fit_function.plotOn(frame, ROOT.RooFit.Components('signal'), ROOT.RooFit.LineColor(ROOT.kRed))
                fit_function.plotOn(frame, ROOT.RooFit.Components('bkg'), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed))

                # compute signal and significance
                mass.setRange('3sigma', mu.getVal() - 3*sigma.getVal(), mu.getVal() + 3*sigma.getVal())

                frac_signal_range = signal.createIntegral(mass_set, mass_norm_set, ROOT.RooFit.Range('3sigma'))
                frac_background_range = background.createIntegral(mass_set, mass_norm_set, ROOT.RooFit.Range('3sigma'))

                sig = n_sig.getVal() * frac_signal_range.getVal()
                sig_error = n_sig.getError() * frac_signal_range.getVal()

                bkg = n_bkg.getVal() * frac_background_range.getVal()
                bkg_error = n_bkg.getError() * frac_background_range.getVal()

                significance = sig / math.sqrt(sig + bkg + 1e-10)
                significance_error = hau.significance_error(sig, bkg)

                # compute chi2
                chi2 = frame.chiSquare('model', 'data', fit_results.floatParsFinal().getSize())

                # add info to plot
                pinfo = ROOT.TPaveText(0.537, 0.474, 0.937, 0.875, 'NDC')
                pinfo.SetBorderSize(0)
                pinfo.SetFillStyle(0)
                pinfo.SetTextAlign(30+3)
                pinfo.SetTextFont(42)
                # pinfo.SetTextSize(12)

                string_list = []
        
                string_list.append('#chi^{2} / NDF ' + f'{chi2:.2f}')
                string_list.append(f'Significance (3 #sigma) {significance:.1f} #pm {significance_error:.1f}')
                string_list.append(f'S (3 #sigma) {sig:.1f} #pm {sig_error:.1f}')
                string_list.append(f'B (3 #sigma) {bkg:.1f} #pm {bkg_error:.1f}')

                for s in string_list:
                    pinfo.AddText(s)

                frame.addObject(pinfo)
                frame.Write()

                # fill the measured hypertriton counts histograms
                fill_raw(model, ctbin, n_sig.getVal(), n_sig.getError(), eff)
                fill_corrected(model, ctbin, n_sig.getVal(), n_sig.getError(), eff)

                if eff == eff_best:
                    fill_raw_best(model, ctbin, n_sig.getVal(), n_sig.getError(), eff)
                    fill_corrected_best(model, ctbin, n_sig.getVal(), n_sig.getError(), eff)


    expo = ROOT.TF1('myexpo', '[0]*exp(-x/([1]*0.029979245800))/([1]*0.029979245800)', 0, 35)
    expo.SetParLimits(1, 100, 5000)

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
        for ctbin_idx in range(len(CT_BINS)-1):
            # random bkg model
            bkg_index = np.random.randint(0, 2)
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
        ct_bins = list(zip(CT_BINS[:-1], CT_BINS[1:]))

        for model, eff in zip(bkg_list, eff_list):
            ctbin = ct_bins[ctbin_idx-1]

            counts, error = get_corrected_counts(model, ctbin, eff)

            tmp_ctdist.SetBinContent(ctbin_idx, counts)
            tmp_ctdist.SetBinError(ctbin_idx, error)

            ctbin_idx += 1

        tmp_ctdist.Fit(expo)

        # # if ct fit is good use it for systematics
        if expo.GetChisquare() > 3 * expo.GetNDF():
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

output_file.Close()
