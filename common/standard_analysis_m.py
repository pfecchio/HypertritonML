#!/usr/bin/env python3
import argparse
import os
import time
import warnings

import numpy as np
import yaml

import hyp_analysis_utils as hau
import hyp_plot_utils as hpu

import ROOT

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

ROOT.gROOT.SetBatch()
ROOT.ROOT.EnableImplicitMT()
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
COLUMNS = params['TRAINING_COLUMNS']

SPLIT_MODE = args.split

SPLIT_CUTS = ['']
SPLIT_LIST = ['']
if SPLIT_MODE:
    SPLIT_LIST = ['_matter','_antimatter']
    SPLIT_CUTS = ['&& ArmenterosAlpha > 0', '&& ArmenterosAlpha < 0']

NBINS = len(PT_BINS) - 1
BINS = np.asarray(PT_BINS, dtype=np.float64)
###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
data_path = os.path.expandvars(params['DATA_PATH'])

BKG_MODELS = params['BKG_MODELS']

results_dir = os.environ['HYPERML_RESULTS_{}'.format(N_BODY)]

###############################################################################
file_name = results_dir + f'/{FILE_PREFIX}_std_results.root'
output_file = ROOT.TFile(file_name, 'recreate')

standard_selection = f'V0CosPA > 0.9999 && NpidClustersHe3 > 80 && He3ProngPt > 1.8 && pt > 2 && pt < 10 && PiProngPt > 0.15 && He3ProngPvDCA > 0.05 && PiProngPvDCA > 0.2 && TPCnSigmaHe3 < 3.5 && TPCnSigmaHe3 > -3.5 && ProngsDCA < 1 && centrality >= {CENT_CLASSES[0][0]} && centrality < {CENT_CLASSES[0][1]} && ct<{CT_BINS[-1]} && ct>{CT_BINS[0]}'

###############################################################################
start_time = time.time()
###############################################################################

###############################################################################
rdf_data = ROOT.RDataFrame('DataTable', data_path)
rdf_mc = ROOT.RDataFrame('SignalTable', signal_path)

mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2.975, 3.01, 'GeV/c^{2}')
mass_set = ROOT.RooArgSet(mass)

delta_mass = ROOT.RooRealVar('delta_m', '#Delta m', -0.005, 0.005, 'GeV/c^{2}')
shift_mass = ROOT.RooAddition('shift_m', "m + #Delta m", ROOT.RooArgList(mass, delta_mass))
###############################################################################


for split, splitcut in zip(SPLIT_LIST, SPLIT_CUTS):
    HIST_MASS = {}
    HIST_SHIFT = {}

    MC_MASS = 2.99131

    for model in BKG_MODELS:
        HIST_MASS[model] = ROOT.TH1D(f'mass_{model}{split}', ';#it{p}_{T} (GeV/#it{c});#it{c}t (cm);m (MeV/#it{c}^{2})', NBINS, BINS)
        HIST_SHIFT[model] = ROOT.TH1D(f'shift_{model}{split}', ';#it{p}_{T} (GeV/#it{c});#it{c}t (cm);m (MeV/#it{c}^{2})', NBINS, BINS)

    df_data = rdf_data.Filter(standard_selection + splitcut)
    df_mc = rdf_mc.Filter(standard_selection + splitcut)

    # define global RooFit objects
    mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2.975, 3.01, 'GeV/c^{2}')
    delta_mass = ROOT.RooRealVar('delta_m', '#Delta m', -0.005, 0.005, 'GeV/c^{2}')
    shift_mass = ROOT.RooAddition('shift_m', "m + #Delta m", ROOT.RooArgList(mass, delta_mass))

    for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
        subdir_name = f'pt{ptbin[0]}{ptbin[1]}'
        pt_dir = output_file.mkdir(subdir_name)
        pt_dir.cd()

        # get data slice as NumPy array
        mass_array_data = df_data.Filter(f'pt<{ptbin[1]} && pt>{ptbin[0]}').AsNumpy(['m'])['m']
        mass_array_mc = df_mc.Filter(f'pt<{ptbin[1]} && pt>{ptbin[0]}').AsNumpy(['m'])['m']

        np.random.shuffle(mass_array_mc)

        roo_data_slice = hau.ndarray2roo(mass_array_data, mass)
        roo_mc_slice = hau.ndarray2roo(mass_array_mc[:50000], mass)

        for model in BKG_MODELS:
            # define signal component
            signal = ROOT.RooKeysPdf('signal', 'signal', shift_mass, mass, roo_mc_slice, ROOT.RooKeysPdf.MirrorBoth, 2.)

            # define background parameters
            slope = ROOT.RooRealVar('slope', 'exponential slope', -100., 100)

            c0 = ROOT.RooRealVar('c0', 'constant c0', -100., 100.)
            c1 = ROOT.RooRealVar('c1', 'constant c1', -100., 100.)
            c2 = ROOT.RooRealVar('c2', 'constant c2', -100., 100.)

            # define background component depending on background model required
            if model == 'pol1':
                background = ROOT.RooPolynomial('bkg', 'pol1 bkg', mass, ROOT.RooArgList(c0, c1))

            if model == 'pol2':
                background = ROOT.RooPolynomial('bkg', 'pol2 bkg', mass, ROOT.RooArgList(c0, c1, c2))

            if model == 'expo':
                background = ROOT.RooExponential('bkg', 'expo bkg', mass, slope)

            # define fraction
            n = ROOT.RooRealVar('n1', 'n1 const', 0., 1, 'GeV')

            # define the fit funciton and perform the actual fit
            fit_function = ROOT.RooAddPdf(f'{model}_gaus', 'signal + background', ROOT.RooArgList(signal, background), ROOT.RooArgList(n))
            fit_results = fit_function.fitTo(roo_data_slice, ROOT.RooFit.Range(2.975, 3.01), ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save())

            frame = mass.frame(35)
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

            ll_frame = delta_mass.frame(ROOT.RooFit.Bins(10), ROOT.RooFit.Range(-0.005, 0.005))

            nll = fit_function.createNLL(roo_data_slice, ROOT.RooFit.NumCPU(8))
            ROOT.RooMinimizer(nll).migrad()

            # pll = nll.createProfile(ROOT.RooArgSet(delta_mass))
 
            nll.plotOn(ll_frame, ROOT.RooFit.ShiftToZero())
            # pll.plotOn(ll_frame, ROOT.RooFit.LineColor(ROOT.kRed))

            ll_frame.SetMinimum(-1)
            ll_frame.Write()

            bin_idx = HIST_MASS[model].FindBin((ptbin[0] + ptbin[1]) / 2)
            
            HIST_MASS[model].SetBinContent(bin_idx, (MC_MASS-delta_mass.getVal())*1000)
            HIST_MASS[model].SetBinError(bin_idx, delta_mass.getError() * 1000)
            
            HIST_SHIFT[model].SetBinContent(bin_idx, delta_mass.getVal()*1000)
            HIST_SHIFT[model].SetBinError(bin_idx, delta_mass.getError() * 1000)


    output_file.cd()
    for model in BKG_MODELS:
        HIST_MASS[model].Write()
        HIST_SHIFT[model].Write()

        hpu.mass_plot_makeup(HIST_MASS[model], model, PT_BINS, split)


###############################################################################
print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')
