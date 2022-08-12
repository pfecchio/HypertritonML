#!/usr/bin/env python3
import argparse
import os
import time

import hyp_analysis_utils as hau
import hyp_plot_utils as hpu
import numpy as np
import pandas as pd
import ROOT
import yaml

import math

ROOT.gROOT.LoadMacro('RooCustomPdfs/RooDSCBShape.cxx++')
# ROOT.gInterpreter.ProcessLine('RooCustomPdfs/RooDSCBShape.h')

from ROOT import RooDSCBShape

kBlueC = ROOT.TColor.GetColor('#1f78b4')
kOrangeC  = ROOT.TColor.GetColor("#ff7f00")

ROOT.gROOT.SetBatch()

np.random.seed(42)

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
parser.add_argument('-s', '--significance',
                    help='Use the BDTefficiency selection from the significance scan', action='store_true')
parser.add_argument('-dbshape', '--dbshape',
                    help='Fit using DSCBShape', action='store_true')

parser.add_argument('-matter', '--matter', help='Run with matter', action='store_true')
parser.add_argument('-antimatter', '--antimatter', help='Run with antimatter', action='store_true')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
###############################################################################

SPLIT = ''

if args.matter:
    SPLIT = '_matter'

if args.antimatter:
    SPLIT = '_antimatter'

###############################################################################

# define some globals
FILE_PREFIX = params['FILE_PREFIX'] + SPLIT


DATA_PATH = os.path.expandvars(params['DATA_PATH'])
MC_PATH = os.path.expandvars(params['MC_PATH'])


BKG_MODELS = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']

CENT_CLASS = params['CENTRALITY_CLASS'][0]
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
EFF_ARRAY = np.around(np.arange(EFF_MIN, EFF_MAX+EFF_STEP, EFF_STEP), 2)

SIGNIFICANCE_SCAN = args.significance
DBSHAPE = args.dbshape

FIX_EFF = 0.70 if not SIGNIFICANCE_SCAN else 0
###############################################################################

###############################################################################
# input/output files
results_dir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]
tables_dir = os.path.dirname(DATA_PATH)
efficiency_dir = os.environ['HYPERML_EFFICIENCIES_{}'.format(params['NBODY'])]

# input data file
file_name = tables_dir + f'/applied_df_{FILE_PREFIX}.parquet.gzip'
data_df = pd.read_parquet(file_name, engine='fastparquet')

# mc file
tables_dir = os.path.dirname(MC_PATH)
file_name = tables_dir + f'/applied_mc_df_{FILE_PREFIX}.parquet.gzip'
print(file_name)
mc_df = pd.read_parquet(file_name, engine='fastparquet')

# significance scan output
file_name = results_dir + f'/Efficiencies/{FILE_PREFIX}_sigscan.npy'
sigscan_dict = np.load(file_name, allow_pickle=True).item()

# output file
suffix = "" if not DBSHAPE else "_dscb"
file_name = results_dir + f'/{FILE_PREFIX}_signal_extraction{suffix}.root'
output_file = ROOT.TFile(file_name, 'recreate')
###############################################################################

###############################################################################
start_time = time.time()
###############################################################################
# define support globals

MASS_H2 = {}
MASS_SHIFT_H2 = {}
RECO_SHIFT_H2 = {}
RAW_COUNTS_H2 = {}


MC_MASS = 2.99131

# prepare histograms for the analysis
for model in BKG_MODELS:
    # initialize histos for the hypertriton counts vs ct vs BDT efficiency
    MASS_H2[model] = ROOT.TH2D(f'mass_{model}', ';#it{c}t cm;BDT efficiency; m (MeV/c^{2})',
                               len(CT_BINS) - 1, np.array(CT_BINS, dtype='double'), len(EFF_ARRAY) - 1, np.array(EFF_ARRAY, dtype='double'))

    RAW_COUNTS_H2[model] = ROOT.TH2D(f'raw_counts_{model}', ';#it{c}t cm;BDT efficiency; Raw Counts',
                               len(CT_BINS) - 1, np.array(CT_BINS, dtype='double'), len(EFF_ARRAY) - 1, np.array(EFF_ARRAY, dtype='double'))

    MASS_SHIFT_H2[model] = MASS_H2[model].Clone(f'mass_shift_{model}')
    RECO_SHIFT_H2[model] = MASS_H2[model].Clone(f'reco_shift_{model}')


# useful methods
def get_eff_index(eff):
    idx = (eff - EFF_MIN + EFF_STEP) * 100
    if isinstance(eff, np.ndarray):
        return idx.astype(int)

    return int(idx)


def get_effscore_dict(ctbin):
    info_string = f'090_210_{ctbin[0]}{ctbin[1]}'
    file_name = efficiency_dir + f'/Eff_Score_{info_string}.npy'

    return {round(e[0], 2): e[1] for e in np.load(file_name).T}

##############################################################################


def fill_histo(histo, ctbin, eff, entry, entry_error):
    bin_idx = histo.FindBin((ctbin[0] + ctbin[1]) / 2, round(eff + 0.005, 3))

    histo.SetBinContent(bin_idx, entry)
    histo.SetBinError(bin_idx, entry_error)


def fill_mu(model, ctbin, eff, mass, mass_error):
    fill_histo(MASS_H2[model], ctbin, eff, mass, mass_error)


def fill_raw_counts(model, ctbin, eff, raw_counts, raw_counts_error):
    fill_histo(RAW_COUNTS_H2[model], ctbin, eff, raw_counts, raw_counts_error)


def fill_mc_mass_shift(model, ctbin, eff, shift, shift_error):
    fill_histo(MASS_SHIFT_H2[model], ctbin, eff, shift, shift_error)


def fill_reco_shift(model, ctbin, eff, shift, shift_error):
    fill_histo(RECO_SHIFT_H2[model], ctbin, eff, shift, shift_error)

###############################################################################


# significance-scan/fixed efficiencies switch
if not SIGNIFICANCE_SCAN:
    eff_best_array = np.full(len(CT_BINS) - 1, FIX_EFF)
else:
    eff_best_array = [round(sigscan_dict[f'ct{ctbin[0]}{ctbin[1]}pt210'][0], 2) for ctbin in zip(
        CT_BINS[:-1], CT_BINS[1:])]

# efficiency ranges for sampling the systematics
syst_eff_ranges = np.asarray(
    [list(range(int(x * 100) - 10, int(x * 100) + 11)) for x in eff_best_array]) / 100
print("RANGES: ", syst_eff_ranges)

eff_best_it = iter(eff_best_array)
eff_range_it = iter(syst_eff_ranges)

###############################################################################
# actual analysis

for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
    score_dict = get_effscore_dict(ctbin)

    # get data slice for this ct bin
    data_slice = data_df.query('@ctbin[0]<ct<@ctbin[1] and 2.960<m<3.040')
    mc_slice = mc_df.query('@ctbin[0]<ct<@ctbin[1] and 2.960<m<3.040')

    ct_dir = output_file.mkdir(f'ct{ctbin[0]}{ctbin[1]}')
    ct_dir.cd()

    eff_best = next(eff_best_it)
    eff_range = next(eff_range_it)

    for eff in eff_range:

        # define global RooFit objects

        mass = ROOT.RooRealVar(
            'm', 'm_{^{3}He+#pi}', 2.960, 3.040, 'GeV/c^{2}')
        mass.setVal(MC_MASS)

        # get the data slice as a RooDataSet
        tsd = score_dict[eff]
        roo_data_slice = hau.ndarray2roo(np.array(data_slice.query(
            'score>@tsd')['m'].values, dtype=np.float64), mass)

        # mc slice for the kde
        mc_array = np.array(mc_slice.query('score>@tsd')['m'].values, dtype=np.float64)

        roo_mc_slice = hau.ndarray2roo(mc_array[:10000], mass)
        # roo_mc_slice_test = hau.ndarray2roo(np.random.choice(mc_array, size=1000, replace=False), mass)

        # kde for the signal component
        if not DBSHAPE:
            delta_mass = ROOT.RooRealVar(
                'delta_m', '#Delta m', -0.0005, 0.0005, 'GeV/c^{2}')
            shift_mass = ROOT.RooAddition(
                'shift_m', "m + #Delta m", ROOT.RooArgList(mass, delta_mass))
            signal = ROOT.RooKeysPdf(
                'signal', 'signal', shift_mass, mass, roo_mc_slice, ROOT.RooKeysPdf.NoMirror, 2.)
            print("Fitting MC with RooKeysPDF")
            fit_results_mc = signal.fitTo(roo_mc_slice, ROOT.RooFit.Range(
                2.960, 3.040), ROOT.RooFit.NumCPU(64), ROOT.RooFit.Save())
            mc_mass_shift = delta_mass.getVal()
            mc_mass_shift_err = delta_mass.getError()

        else:
            mu = ROOT.RooRealVar('mu', 'hypertriton mass',
                                 2.989, 2.993, 'GeV/c^{2}')
            sigma = ROOT.RooRealVar(
                'sigma', 'hypertriton width', 0.0001, 0.004, 'GeV/c^{2}')
            a1 = ROOT.RooRealVar('a1', 'a1', 0, 5.)
            a2 = ROOT.RooRealVar('a2', 'a2', 0, 10.)
            n1 = ROOT.RooRealVar('n1', 'n1', 1, 10.)
            n2 = ROOT.RooRealVar('n2', 'n2', 1, 10.)
            signal = ROOT.RooDSCBShape(
                'cb', 'cb', mass, mu, sigma, a1, n1, a2, n2)
            fit_results_mc_dscb = signal.fitTo(roo_mc_slice, ROOT.RooFit.Range(
                2.960, 3.040), ROOT.RooFit.NumCPU(64), ROOT.RooFit.Save())
            reco_shift = MC_MASS - mu.getVal()
            reco_shift_err = mu.getError()

            sigma.setConstant(ROOT.kTRUE)
            a1.setConstant(ROOT.kTRUE)
            a2.setConstant(ROOT.kTRUE)
            n1.setConstant(ROOT.kTRUE)
            n2.setConstant(ROOT.kTRUE)

        # fit the kde to the MC for systematic estimate

        frame = mass.frame(80)
        frame.SetName(f'mc_eff{eff:.2f}_{model}')

        roo_mc_slice.plotOn(frame, ROOT.RooFit.Name('MC'))
        signal.plotOn(frame, ROOT.RooFit.Name('signal pdf'),
                      ROOT.RooFit.LineColor(ROOT.kBlue))

        # add info to plot
        pinfo = ROOT.TPaveText(0.537, 0.474, 0.937, 0.875, 'NDC')
        pinfo.SetBorderSize(0)
        pinfo.SetFillStyle(0)
        pinfo.SetTextAlign(30+3)
        pinfo.SetTextFont(42)
        frame.addObject(pinfo)
        frame.Write()

        # loop over the possible background models
        for model in BKG_MODELS:
            # define background parameters
            slope = ROOT.RooRealVar('slope', 'exponential slope', -100., 100.)

            c0 = ROOT.RooRealVar('c0', 'constant c0', -1., 1.)
            c1 = ROOT.RooRealVar('c1', 'constant c1', -1., 1.)

            # define background component depending on background model required
            if model == 'pol1':
                background = ROOT.RooPolynomial(
                    'bkg', 'pol1 bkg', mass, ROOT.RooArgList(c0))

            if model == 'pol2':
                background = ROOT.RooPolynomial(
                    'bkg', 'pol2 bkg', mass, ROOT.RooArgList(c0, c1))

            if model == 'expo':
                background = ROOT.RooExponential(
                    'bkg', 'expo bkg', mass, slope)

            # define fraction
            n = ROOT.RooRealVar('n', 'n const', 0., 1, 'GeV')

            # define the fit funciton and perform the actual fit
            fit_function = ROOT.RooAddPdf(
                f'{model}_total_pdf', 'signal + background', ROOT.RooArgList(signal, background), ROOT.RooArgList(n))
            fit_results = fit_function.fitTo(roo_data_slice, ROOT.RooFit.Range(
                2.960, 3.040), ROOT.RooFit.NumCPU(64), ROOT.RooFit.Save())

            frame = mass.frame(80)
            frame.SetName(f'eff{eff:.2f}_{model}')

            roo_data_slice.plotOn(frame, ROOT.RooFit.Name('data'), ROOT.RooFit.MarkerSize(1.5))
            fit_function.plotOn(frame, ROOT.RooFit.Components('bkg'), ROOT.RooFit.LineStyle(9), ROOT.RooFit.LineColor(kOrangeC))
            fit_function.plotOn(frame, ROOT.RooFit.LineColor(kBlueC))

            signal_counts = n.getVal()*roo_data_slice.sumEntries()
            signal_counts_error = (n.getError()/n.getVal())*n.getVal()*roo_data_slice.sumEntries()

            fill_raw_counts(model, ctbin, eff, signal_counts, signal_counts_error)

            if not DBSHAPE:
                m =  (MC_MASS-delta_mass.getVal())* 1e3
                m_error = delta_mass.getError()*1e3
                significance, significance_error = hau.compute_significance(roo_data_slice, mass, signal, background, n, mu = None, sigma = None)
                fill_mu(model, ctbin, eff, m, m_error)
                fill_mc_mass_shift(model, ctbin, eff, mc_mass_shift*1e3, mc_mass_shift_err*1e3)                
            else:
                m =  mu.getVal() * 1e3
                m_error = mu.getError()*1e3
                significance, significance_error = hau.compute_significance(roo_data_slice, mass, signal, background, n, mu = mu, sigma = sigma)

                fill_mu(model, ctbin, eff, m , m_error)
                fill_reco_shift(model, ctbin, eff, reco_shift *1e3, reco_shift_err*1e3)


            # compute chi2
            chi2 = frame.chiSquare(
                'model', 'data', fit_results.floatParsFinal().getSize())
            # add info to plot
            pinfo = ROOT.TPaveText(0.737, 0.674, 0.937, 0.875, 'NDC')
            pinfo.SetBorderSize(0)
            pinfo.SetFillStyle(0)
            pinfo.SetTextAlign(30+3)
            pinfo.SetTextFont(42)
            string_list = []
            string_list.append('#chi^{2} / NDF ' + f'{chi2:.2f}')
            string_list.append(f'S (3 #sigma) {signal_counts:.1f} #pm {signal_counts_error:.1f}')
            string_list.append(f'Significance (3 #sigma) {significance:.1f} #pm {significance_error:.1f}')
            string_list.append('m_{ {}^{3}_{#Lambda}H} = ' + f'{m:.3f} #pm {m_error:.3f}')
            for s in string_list:
                pinfo.AddText(s)
            frame.addObject(pinfo)
            frame.Write()


output_file.cd()
for model in BKG_MODELS:
    MASS_H2[model].Write()
    RAW_COUNTS_H2[model].Write()
    MASS_SHIFT_H2[model].Write() if not DBSHAPE else RECO_SHIFT_H2[model].Write()

output_file.Close()
###############################################################################
print(
    f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')
