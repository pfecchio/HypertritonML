#!/usr/bin/env python3
import argparse
import os
import time
import warnings

import numpy as np
import yaml

import hyp_analysis_utils as hau
import hyp_plot_utils as hpu

import math

import ROOT

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

ROOT.gROOT.SetBatch()
ROOT.ROOT.EnableImplicitMT(32)
ROOT.RooMsgService.instance().setSilentMode(True)

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('config', help='Path to the YAML configuration file')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
###############################################################################

###############################################################################
# define analysis global variables
N_BODY = params['NBODY']
FILE_PREFIX = params['FILE_PREFIX']

CENT_CLASSES = params['CENTRALITY_CLASS']
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']

SPLIT_MODE = args.split

SPLIT_CUTS = ['']
SPLIT_LIST = ['']
if SPLIT_MODE:
    SPLIT_LIST = ['_matter','_antimatter']
    SPLIT_CUTS = ['&& ArmenterosAlpha > 0', '&& ArmenterosAlpha < 0']

NBINS = len(CT_BINS) - 1
BINS = np.asarray(CT_BINS, dtype=np.float64)

KDE_SAMPLE_SIZE = 20000
BANDWIDTH = 1.905

MC_MASS = 2.99131

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
data_path = os.path.expandvars(params['DATA_PATH'])

BKG_MODELS = params['BKG_MODELS']

results_dir = os.environ['HYPERML_RESULTS_{}'.format(N_BODY)]
efficiency_dir = os.environ['HYPERML_EFFICIENCIES_{}'.format(params['NBODY'])]
utils_dir = os.environ['HYPERML_UTILS_2']

# preselection eff
file_name = efficiency_dir + f'/{FILE_PREFIX}_preseleff_cent090.root'
efficiency_file = ROOT.TFile(file_name, 'read')
EFFICIENCY = efficiency_file.Get('PreselEff').ProjectionY()

# absorption correction file
file_name = utils_dir + '/he3abs/recCtHe3.root'
abs_file = ROOT.TFile(file_name, 'read')
ABSORPTION = abs_file.Get('Reconstructed ct spectrum')

###############################################################################
file_name = results_dir + f'/{FILE_PREFIX}_std_results.root'
output_file = ROOT.TFile(file_name, 'recreate')

standard_selection = f'V0CosPA > 0.9999 && NpidClustersHe3 > 80 && He3ProngPt > 1.8 && pt > 2 && pt < 10 && PiProngPt > 0.15 && He3ProngPvDCA > 0.05 && PiProngPvDCA > 0.2 && TPCnSigmaHe3 < 3.5 && TPCnSigmaHe3 > -3.5 && ProngsDCA < 1 && centrality >= {CENT_CLASSES[0][0]} && centrality < {CENT_CLASSES[0][1]} && ct<{CT_BINS[-1]} && ct>{CT_BINS[0]}'

###############################################################################
start_time = time.time()
###############################################################################
for split in SPLIT_LIST:
    HIST_COUNTS = {}
    for model in BKG_MODELS:
        HIST_COUNTS[model] = ROOT.TH1D(f'counts_{model}{split}', '#it{c}t (cm);m (MeV/#it{c}^{2})', NBINS, BINS)


def get_presel_eff(ctbin):
    return EFFICIENCY.GetBinContent(EFFICIENCY.FindBin((ctbin[0] + ctbin[1]) / 2))


def get_absorption_correction(ctbin):
    bin_idx = ABSORPTION.FindBin((ctbin[0] + ctbin[1]) / 2)
    return 1 - ABSORPTION.GetBinContent(bin_idx)


def fill_corrected_counts(bkg, ctbin, counts, counts_err):
    bin_idx = HIST_COUNTS[bkg].FindBin((ctbin[0] + ctbin[1]) / 2)

    abs_corr = get_absorption_correction(ctbin)
    presel_eff = get_presel_eff(ctbin)
    bin_width = HIST_COUNTS[bkg].GetBinWidth(bin_idx)

    HIST_COUNTS[bkg].SetBinContent(bin_idx, counts/presel_eff/abs_corr/bin_width)
    HIST_COUNTS[bkg].SetBinError(bin_idx, counts_err/presel_eff/abs_corr/bin_width)

###############################################################################
rdf_data = ROOT.RDataFrame('DataTable', data_path)
rdf_mc = ROOT.RDataFrame('SignalTable', signal_path)

mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2.960, 3.040, 'GeV/c^{2}')
mass_set = ROOT.RooArgSet(mass)
mass_norm_set = ROOT.RooFit.NormSet(mass_set)

delta_mass = ROOT.RooRealVar('delta_m', '#Delta m', -0.005, 0.005, 'GeV/c^{2}')
shift_mass = ROOT.RooAddition('shift_m', "m + #Delta m", ROOT.RooArgList(mass, delta_mass))
###############################################################################

for split, splitcut in zip(SPLIT_LIST, SPLIT_CUTS):
    HIST_COUNTS = {}

    for model in BKG_MODELS:
        HIST_COUNTS[model] = ROOT.TH1D(f'counts_{model}{split}', '#it{c}t (cm);m (MeV/#it{c}^{2})', NBINS, BINS)

    df_data = rdf_data.Filter(standard_selection + splitcut)
    df_mc = rdf_mc.Filter(standard_selection + splitcut)

    # define global RooFit objects
    mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2.975, 3.01, 'GeV/c^{2}')
    delta_mass = ROOT.RooRealVar('delta_m', '#Delta m', -0.005, 0.005, 'GeV/c^{2}')
    shift_mass = ROOT.RooAddition('shift_m', "m + #Delta m", ROOT.RooArgList(mass, delta_mass))

    for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
        subdir_name = f'ct{ctbin[0]}{ctbin[1]}'
        pt_dir = output_file.mkdir(subdir_name)
        pt_dir.cd()

        # get data slice as NumPy array
        mass_array_data = df_data.Filter(f'ct<{ctbin[1]} && ct>{ctbin[0]}').AsNumpy(['m'])['m']
        mass_array_mc = df_mc.Filter(f'ct<{ctbin[1]} && ct>{ctbin[0]}').AsNumpy(['m'])['m']

        np.random.shuffle(mass_array_mc)

        roo_data_slice = hau.ndarray2roo(mass_array_data, mass)
        roo_mc_slice = hau.ndarray2roo(mass_array_mc[:KDE_SAMPLE_SIZE], mass)

        for model in BKG_MODELS:
            # define signal parameters
            mu = ROOT.RooRealVar('mu', 'hypertriton mass', 2.989, 2.993, 'GeV/c^{2}')
            sigma = ROOT.RooRealVar('sigma', 'hypertriton width', 0.0001, 0.004, 'GeV/c^{2}')

            # define temporary signal component for signal region determination
            signal_tmp = ROOT.RooGaussian('signal', 'signal component pdf', mass, mu, sigma)

            # define background parameters
            slope = ROOT.RooRealVar('slope', 'exponential slope', -100., 100)

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
            fit_results_tmp = fit_function_tmp.fitTo(roo_data_slice, ROOT.RooFit.Range(2.960, 3.040), ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save())

            mu.setConstant(ROOT.kTRUE)
            mu.removeError()
            sigma.setConstant(ROOT.kTRUE)
            sigma.removeError()
                
            # define the signal component
            mass.setVal(MC_MASS)
            signal = ROOT.RooKeysPdf('signal', 'signal', shift_mass, mass, roo_mc_slice, ROOT.RooKeysPdf.NoMirror, BANDWIDTH)

            # define the fit funciton -> signal component + background component
            fit_function = ROOT.RooAddPdf(f'{model}_gaus', 'signal + background', ROOT.RooArgList(signal, background), ROOT.RooArgList(n_sig, n_bkg))

            fit_results = fit_function.fitTo(roo_data_slice, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.Range(2.960, 3.040), ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save())

            frame = mass.frame(60)
            frame.SetName(model)

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

            fill_corrected_counts(model, ctbin, sig, sig_error)

    output_file.cd()

    expo = ROOT.TF1('myexpo', '[0]*exp(-x/([1]*0.029979245800))/([1]*0.029979245800)', 0, 35)
    expo.SetParLimits(1, 100, 5000)

    kBlueC = ROOT.TColor.GetColor('#1f78b4')
    kBlueCT = ROOT.TColor.GetColorTransparent(kBlueC, 0.5)
    kRedC = ROOT.TColor.GetColor('#e31a1c')
    kRedCT = ROOT.TColor.GetColorTransparent(kRedC, 0.5)

    for model in BKG_MODELS:
        output_file.cd()

        HIST_COUNTS[model].Write()


        HIST_COUNTS[model].UseCurrentStyle()
        HIST_COUNTS[model].Fit(expo, 'MEI0+', '', 0, 35)

        fit_function = HIST_COUNTS[model].GetFunction('myexpo')
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
        HIST_COUNTS[model].Draw('ex0same')
        HIST_COUNTS[model].SetMarkerStyle(20)
        HIST_COUNTS[model].SetMarkerColor(kBlueC)
        HIST_COUNTS[model].SetLineColor(kBlueC)
        HIST_COUNTS[model].SetMinimum(0.001)
        HIST_COUNTS[model].SetMaximum(1000)
        HIST_COUNTS[model].SetStats(0)

        frame.GetYaxis().SetTitleSize(26)
        frame.GetYaxis().SetLabelSize(22)
        frame.GetXaxis().SetTitleSize(26)
        frame.GetXaxis().SetLabelSize(22)
        frame.GetYaxis().SetRangeUser(7, 5000)
        frame.GetXaxis().SetRangeUser(0.5, 35.5)

        pinfo.Draw('x0same')

        canvas.Write()

###############################################################################
print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')
