#!/usr/bin/env python3
import argparse
import os
import time
import warnings

import numpy as np
import yaml

import hyp_analysis_utils as hau
import hyp_plot_utils as hpu
import pandas as pd
import uproot

import ROOT

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

ROOT.gROOT.SetBatch()
ROOT.RooMsgService.instance().setSilentMode(True)

np.random.seed(42)

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
COLUMNS = params['TRAINING_COLUMNS']

SPLIT_MODE = args.split

SPLIT_CUTS = ['']
SPLIT_LIST = ['']
if SPLIT_MODE:
    SPLIT_LIST = ['_matter','_antimatter']
    SPLIT_CUTS = ['and ArmenterosAlpha > 0', 'and ArmenterosAlpha < 0']

NBINS = len(CT_BINS) - 1
BINS = np.asarray(CT_BINS, dtype=np.float64)

MC_MASS = 2.99131

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
data_path = os.path.expandvars(params['DATA_PATH'])

BKG_MODELS = params['BKG_MODELS']

results_dir = os.environ['HYPERML_RESULTS_{}'.format(N_BODY)]

###############################################################################
file_name = results_dir + f'/{FILE_PREFIX}_std_results.root'
output_file = ROOT.TFile(file_name, 'recreate')

standard_selection = f'V0CosPA > 0.9999 and NpidClustersHe3 > 80 and He3ProngPt > 1.8 and pt > 2 and pt < 10 and PiProngPt > 0.15 and He3ProngPvDCA > 0.05 and PiProngPvDCA > 0.2 and TPCnSigmaHe3 < 3.5 and TPCnSigmaHe3 > -3.5 and ProngsDCA < 1 and centrality >= {CENT_CLASSES[0][0]} and centrality < {CENT_CLASSES[0][1]} and ct<{CT_BINS[-1]} and ct>{CT_BINS[0]}'

###############################################################################
start_time = time.time()
###############################################################################

###############################################################################
# print(data_path)
rdf_data = pd.read_parquet(data_path, engine='fastparquet')

rdf_mc = uproot.open(signal_path)['SignalTable'].arrays(library='pd')

mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2.960, 3.040, 'GeV/c^{2}')
mass_set = ROOT.RooArgSet(mass)

delta_mass = ROOT.RooRealVar('delta_m', '#Delta m', -0.0005, 0.0005, 'GeV/c^{2}')
shift_mass = ROOT.RooAddition('shift_m', "m + #Delta m", ROOT.RooArgList(mass, delta_mass))
###############################################################################


for split, splitcut in zip(SPLIT_LIST, SPLIT_CUTS):
    HIST_MASS = {}
    HIST_SHIFT = {}

    for model in BKG_MODELS:
        HIST_MASS[model] = ROOT.TH1D(f'mass_{model}{split}', ';#it{p}_{T} (GeV/#it{c});#it{c}t (cm);m (MeV/#it{c}^{2})', NBINS, BINS)
        HIST_SHIFT[model] = ROOT.TH1D(f'shift_{model}{split}', ';#it{p}_{T} (GeV/#it{c});#it{c}t (cm);m (MeV/#it{c}^{2})', NBINS, BINS)

    df_data = rdf_data.query(standard_selection + splitcut)
    df_mc = rdf_mc.query(standard_selection + splitcut)

    for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
        subdir_name = f'ct{ctbin[0]}{ctbin[1]}'
        ct_dir = output_file.mkdir(subdir_name)
        ct_dir.cd()

        # get data slice as NumPy array
        mass_array_data = np.array(df_data.query(f'ct<{ctbin[1]} and ct>{ctbin[0]} and m>2.960 and m<3.040')['m'])
        mass_array_mc = np.array(df_mc.query(f'ct<{ctbin[1]} and ct>{ctbin[0]} and m>2.960 and m<3.040')['m'])
        print(mass_array_mc)

        roo_data_slice = hau.ndarray2roo(mass_array_data, mass)
        roo_mc_slice = hau.ndarray2roo(mass_array_mc[:1000], mass)

        # define signal component
        signal = ROOT.RooKeysPdf('signal', 'signal', shift_mass, mass, roo_mc_slice, ROOT.RooKeysPdf.NoMirror, 2.)

        # fit the kde to the MC for systematic estimate
        fit_results_mc = signal.fitTo(roo_mc_slice, ROOT.RooFit.Range(2.960, 3.040), ROOT.RooFit.NumCPU(32), ROOT.RooFit.Save())

        mass_shift = delta_mass.getVal()
        mass_shift_err = delta_mass.getError()

        for model in BKG_MODELS:
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
            n = ROOT.RooRealVar('n1', 'n1 const', 0., 1, 'GeV')

            # define the fit funciton and perform the actual fit
            fit_function = ROOT.RooAddPdf(f'{model}_gaus', 'signal + background', ROOT.RooArgList(signal, background), ROOT.RooArgList(n))
            fit_results = fit_function.fitTo(roo_data_slice, ROOT.RooFit.Range(2.960, 3.040), ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save())

            frame = mass.frame(70)
            frame.SetName(f'{model}')

            roo_data_slice.plotOn(frame, ROOT.RooFit.Name('data'))
            fit_function.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Name('model'))
            fit_function.plotOn(frame, ROOT.RooFit.Components('signal'), ROOT.RooFit.LineStyle(ROOT.kDotted), ROOT.RooFit.LineColor(ROOT.kRed))
            fit_function.plotOn(frame, ROOT.RooFit.Components('bkg'), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed))

            # compute chi2
            chi2 = frame.chiSquare('model', 'data', fit_results.floatParsFinal().getSize())

            # add info to plot
            pinfo = ROOT.TPaveText(0.537, 0.474, 0.937, 0.875, 'NDC')
            pinfo.SetBorderSize(0)
            pinfo.SetFillStyle(0)
            pinfo.SetTextAlign(30+3)
            pinfo.SetTextFont(42)

            string_list = []
        
            string_list.append('#chi^{2} / NDF ' + f'{chi2:.2f}')
            string_list.append(f'#Delta m = {delta_mass.getVal()*1e6:.1f} #pm {delta_mass.getError()*1e6:.1f} keV')

            for s in string_list:
                pinfo.AddText(s)

            frame.addObject(pinfo)
            frame.Write()

            bin_idx = HIST_MASS[model].FindBin((ctbin[0] + ctbin[1]) / 2)
            
            HIST_MASS[model].SetBinContent(bin_idx, (MC_MASS-delta_mass.getVal()+mass_shift)*1000)
            HIST_MASS[model].SetBinError(bin_idx, delta_mass.getError() * 1000)
            
            # HIST_SHIFT[model].SetBinContent(bin_idx, delta_mass.getVal()*1000)
            # HIST_SHIFT[model].SetBinError(bin_idx, delta_mass.getError() * 1000)


    output_file.cd()
    for model in BKG_MODELS:
        HIST_MASS[model].Write()
        HIST_SHIFT[model].Write()

        hpu.mass_plot_makeup(HIST_MASS[model], model, CT_BINS, split)


###############################################################################
print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')
