#!/usr/bin/env python3

import argparse
import math
import os
import time

import hyp_analysis_utils as hau
import hyp_plot_utils as hpu
import numpy as np
import pandas as pd
import ROOT
import uproot
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

SPLIT_LIST = ['_matter','_antimatter'] if False else ['']
BKG_MODELS = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']

CENT_CLASS = params['CENTRALITY_CLASS'][0]
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
EFF_ARRAY = np.around(np.arange(EFF_MIN, EFF_MAX+0.01, EFF_STEP), 2)

SPLIT_MODE = False
SIGNIFICANCE_SCAN = args.significance

SKIP_FITS = args.skipfits
SYSTEMATICS = True if SKIP_FITS else args.systematics

SYSTEMATICS_COUNTS = 100000

FIX_EFF = 0.70 if not SIGNIFICANCE_SCAN else 0
###############################################################################

###############################################################################
# input/output files
results_dir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]
tables_dir = os.path.dirname(DATA_PATH)
efficiency_dir = os.environ['HYPERML_EFFICIENCIES_{}'.format(params['NBODY'])]

# input data file
file_name = tables_dir + f'/applied_df_{FILE_PREFIX}.parquet.gzip'
data_df = pd.read_parquet(file_name)

# mc file
file_name = tables_dir + f'/applied_mc_df_{FILE_PREFIX}.parquet.gzip'
mc_df = pd.read_parquet(file_name)

# significance scan output
file_name = results_dir + f'/Efficiencies/{FILE_PREFIX}_sigscan.npy'
sigscan_dict = np.load(file_name, allow_pickle=True).item()

# output file
file_name = results_dir + f'/{FILE_PREFIX}_blambda.root'
output_file = ROOT.TFile(file_name, 'update' if SKIP_FITS else 'recreate')
###############################################################################

###############################################################################
start_time = time.time()
###############################################################################

###############################################################################
# define support globals and methods for getting hypertriton counts
MASS_H2 = {}
MASS_BEST = {}

MASS_SHIFT_H2 = {}
MASS_SHIFT_BEST = {}

MC_MASS = 2.99131


for split in SPLIT_LIST:
    for model in BKG_MODELS:
        # initialize histos for the hypertriton counts vs ct vs BDT efficiency
        if SKIP_FITS:
            MASS_H2[model] = output_file.Get(f'mass_{model}')
        else:
            MASS_H2[model] = ROOT.TH2D(f'mass_{model}{split}', ';#it{p}_{T} (GeV/#it{c});BDT efficiency; m (MeV/c^{2})',
            len(PT_BINS) - 1, np.array(PT_BINS, dtype='double'), len(EFF_ARRAY) - 1, np.array(EFF_ARRAY, dtype='double'))
            MASS_BEST[model] = MASS_H2[model].ProjectionX(f'mass_best_{model}')

            MASS_SHIFT_H2[model] = MASS_H2[model].Clone(f'mass_shift_{model}{split}')
            MASS_SHIFT_BEST[model] = MASS_BEST[model].Clone(f'mass_shift_best_{model}{split}')


def get_eff_index(eff):
    idx = (eff - EFF_MIN + EFF_STEP) * 100

    if isinstance(eff, np.ndarray):
        return idx.astype(int)

    return int(idx)


def get_effscore_dict(ptbin):
    info_string = f'090_{ptbin[0]}{ptbin[1]}_090'
    file_name = efficiency_dir + f'/Eff_Score_{info_string}.npy'

    return {round(e[0], 2): e[1] for e in np.load(file_name).T}

def fill_histo(histo, ptbin, eff, entry, entry_error):
    bin_idx = histo.FindBin((ptbin[0] + ptbin[1]) / 2, round(eff + 0.005, 3))

    histo.SetBinContent(bin_idx, entry)
    histo.SetBinError(bin_idx, entry_error)


def fill_histo_best(histo, ptbin, entry, entry_error):
    bin_idx = histo.FindBin((ptbin[0] + ptbin[1]) / 2)

    histo.SetBinContent(bin_idx, entry)
    histo.SetBinError(bin_idx, entry_error)


def fill_mu(model, ptbin, eff, mass, mass_error):
    fill_histo(MASS_H2[model], ptbin, eff, mass, mass_error)
    

def fill_mu_best(model, ptbin, mass, mass_error):
    fill_histo_best(MASS_BEST[model], ptbin, mass, mass_error)


def fill_shift(model, ptbin, eff, shift, shift_error):
    fill_histo(MASS_SHIFT_H2[model], ptbin, eff, shift, shift_error)
    

def fill_shift_best(model, ptbin, shift, shift_error):
    fill_histo_best(MASS_SHIFT_BEST[model], ptbin, shift, shift_error)


def get_measured_mass(bkg, ptbin, eff):
    bin_idx = MASS_H2[bkg].FindBin((ptbin[0] + ptbin[1]) / 2, eff + 0.005)

    mass = MASS_H2[bkg].GetBinContent(bin_idx)
    error = MASS_H2[bkg].GetBinError(bin_idx)

    return mass, error


# define function for fit result plot
def plot_fit(var, data, ptbin):
    var_set = ROOT.RooArgSet(var)
    var_norm_set = ROOT.RooFit.NormSet(var_set)

    frame = mass.frame(60)
    frame.SetName(f'ct{ctbin[0]}{ctbin[1]}_eff{eff:.2f}_{model}')

    data.plotOn(frame, ROOT.RooFit.Name('data'))
    fit_function.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Name('model'))
    fit_function.plotOn(frame, ROOT.RooFit.Components('signal'), ROOT.RooFit.LineStyle(ROOT.kDotted), ROOT.RooFit.LineColor(ROOT.kRed))
    fit_function.plotOn(frame, ROOT.RooFit.Components('bkg'), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed))

    # compute signal and significance
    mass.setRange('3sigma', mu.getVal() - 3*sigma.getVal(), mu.getVal() + 3*sigma.getVal())
    
    frac_signal_range = signal.createIntegral(var_set, var_norm_set, ROOT.RooFit.Range('3sigma'))
    frac_background_range = background[model].createIntegral(var_set, var_norm_set, ROOT.RooFit.Range('3sigma'))

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

    sig = n_sig.getVal()
    sig_error = n_sig.getError()

    return sig, sig_error
###############################################################################

# significance-scan/fixed efficiencies switch
if not SIGNIFICANCE_SCAN:
    eff_best_array = np.full(len(PT_BINS) - 1, FIX_EFF)
else:
    eff_best_array = [round(sigscan_dict[f'ct090pt{ptbin[0]}{ptbin[1]}'][0], 2) for ptbin in zip(PT_BINS[:-1], PT_BINS[1:])]

# efficiency ranges for sampling the systematics
syst_eff_ranges = [list(range(int(x * 100) - 10, int(x * 100) + 11)) for x in eff_best_array]

eff_best_it = iter(eff_best_array)

# actual analysis
for split in SPLIT_LIST:
    if SKIP_FITS:
        continue

    for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
        score_dict = get_effscore_dict(ptbin)

        # get data slice for this ct bin
        data_slice = data_df.query('@ptbin[0]<pt<@ptbin[1]')
        mc_slice = mc_df.query('@ptbin[0]<pt<@ptbin[1]')

        subdir_name = f'pt{ptbin[0]}{ptbin[1]}'
        pt_dir = output_file.mkdir(subdir_name)
        pt_dir.cd()

        eff_best = next(eff_best_it)
        for eff in EFF_ARRAY:
            # define global RooFit objects
            mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2.975, 3.01, 'GeV/c^{2}')
            delta_mass = ROOT.RooRealVar('delta_m', '#Delta m', -0.005, 0.005, 'GeV/c^{2}')
            shift_mass = ROOT.RooAddition('shift_m', "m + #Delta m", ROOT.RooArgList(mass, delta_mass))
               
            # get the data slice as a RooDataSet
            tsd = score_dict[eff]
            roo_data_slice = hau.ndarray2roo(np.array(data_slice.query('score>@tsd')['m'].values, dtype=np.float64), mass)

            mc_array = np.array(mc_slice.query('score>@tsd')['m'].values, dtype=np.float64)
            np.random.shuffle(mc_array)

            roo_mc_slice = hau.ndarray2roo(mc_array[:10000], mass)

            for model in BKG_MODELS:
                # define signal component
                signal = ROOT.RooKeysPdf('signal', 'signal', shift_mass, mass, roo_mc_slice, ROOT.RooKeysPdf.NoMirror, 2.)

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
                frame.SetName(f'eff{eff:.2f}_{model}')

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

                # ll_frame = delta_mass.frame(ROOT.RooFit.Bins(10), ROOT.RooFit.Range(-0.005, 0.005))

                # nll = fit_function.createNLL(roo_data_slice, ROOT.RooFit.NumCPU(8))
                # ROOT.RooMinimizer(nll).migrad()

                # pll = nll.createProfile(ROOT.RooArgSet(delta_mass))
 
                # nll.plotOn(ll_frame, ROOT.RooFit.ShiftToZero())
                # pll.plotOn(ll_frame, ROOT.RooFit.LineColor(ROOT.kRed))

                # ll_frame.SetMinimum(-1)
                # ll_frame.Write()

                fill_mu(model, ptbin, eff, (MC_MASS-delta_mass.getVal())*1000, delta_mass.getError()*1000)
                fill_shift(model, ptbin, eff, delta_mass.getVal()*1000, delta_mass.getError()*1000)
                if eff == eff_best:
                    fill_mu_best(model, ptbin, (MC_MASS-delta_mass.getVal())*1000, delta_mass.getError()*1000)
                    fill_shift_best(model, ptbin, delta_mass.getVal()*1000, delta_mass.getError()*1000)

    output_file.cd()
    for model in BKG_MODELS:
        MASS_H2[model].Write()
        MASS_BEST[model].Write()

        MASS_SHIFT_H2[model].Write()
        MASS_SHIFT_BEST[model].Write()

        hpu.mass_plot_makeup(MASS_BEST[model], model, PT_BINS, split)


if SYSTEMATICS:
    # systematics histos
    blambda_dist = ROOT.TH1D(f'syst_blambda{split}', ';B_{#Lambda} MeV ;counts', 100, -0.5, 0.5)

    tmp_mass = MASS_H2[BKG_MODELS[0]].ProjectionX('tmp_mass')

    combinations = set()

    for _ in range(SYSTEMATICS_COUNTS):
        tmp_mass.Reset()

        bkg_list = []
        eff_list = []
        bkg_idx_list = []
        eff_idx_list = []

        # loop over ptbins
        for ptbin_idx in range(len(PT_BINS)-1):
            # random bkg model
            bkg_index = np.random.randint(0, len(BKG_MODELS))
            bkg_idx_list.append(bkg_index)
            bkg_list.append(BKG_MODELS[bkg_index])

            # randon BDT efficiency in the defined range
            eff = np.random.choice(syst_eff_ranges[ptbin_idx]) / 100
            eff_list.append(eff)
            eff_index = get_eff_index(eff)
            eff_idx_list.append(eff_index)

        # convert indexes into hash and if already sampled skip this combination
        combo = ''.join(map(str, bkg_idx_list + eff_idx_list))
        if combo in combinations:
            continue

        # if indexes are good measure B_{Lambda}
        ptbin_idx = 1
        pt_bins = list(zip(PT_BINS[:-1], PT_BINS[1:]))

        mass_list = []

        for model, eff in zip(bkg_list, eff_list):
            ptbin = pt_bins[ptbin_idx - 1]

            mass, mass_error = get_measured_mass(model, ptbin, eff)

            tmp_mass.SetBinContent(ptbin_idx, mass)
            tmp_mass.SetBinError(ptbin_idx, mass_error)

            mass_list.append(mass)

            ptbin_idx += 1

        wmass, _ = hau.histo_weighted_mean(tmp_mass)
        blambda = 1115.683 + 1875.61294257 - wmass

        blambda_dist.Fill(blambda)
        combinations.add(combo)

    blambda_dist.Write()

###############################################################################
print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')
