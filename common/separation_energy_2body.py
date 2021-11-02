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

ROOT.gSystem.Load('RooCustomPdfs/libRooDSCBShape.so')
from ROOT import RooDSCBShape

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
EFF_ARRAY = np.around(np.arange(EFF_MIN, EFF_MAX+EFF_STEP, EFF_STEP), 2)

SPLIT_MODE = False
SIGNIFICANCE_SCAN = args.significance

SKIP_FITS = args.skipfits
SYSTEMATICS = True if SKIP_FITS else args.systematics

SYSTEMATICS_COUNTS = 10000

FIX_EFF = 0.80 if not SIGNIFICANCE_SCAN else 0
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
file_name = tables_dir + f'/applied_mc_df_{FILE_PREFIX}.parquet.gzip'
mc_df = pd.read_parquet(file_name, engine='fastparquet')

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
# define support globals 
MASS_H2 = {}
MASS_BEST = {}

MASS_SHIFT_H2 = {}
MASS_SHIFT_BEST = {}

MC_MASS = 2.99131

# prepare histograms for the analysis
for split in SPLIT_LIST:
    for model in BKG_MODELS:
        # initialize histos for the hypertriton counts vs ct vs BDT efficiency
        if SKIP_FITS:
            MASS_H2[model] = output_file.Get(f'mass_{model}{split}')
            
        else:
            MASS_H2[model] = ROOT.TH2D(f'mass_{model}{split}', ';#it{c}t cm;BDT efficiency; m (MeV/c^{2})',
            len(CT_BINS) - 1, np.array(CT_BINS, dtype='double'), len(EFF_ARRAY) - 1, np.array(EFF_ARRAY, dtype='double'))
            MASS_BEST[model] = MASS_H2[model].ProjectionX(f'mass_best_{model}{split}')

            MASS_SHIFT_H2[model] = MASS_H2[model].Clone(f'mass_shift_{model}{split}')
            MASS_SHIFT_BEST[model] = MASS_BEST[model].Clone(f'mass_shift_best_{model}{split}')


# usefull methods
def get_eff_index(eff):
    idx = (eff - EFF_MIN + EFF_STEP) * 100
    if isinstance(eff, np.ndarray):
        return idx.astype(int)

    return int(idx)


def get_effscore_dict(ctbin):
    info_string = f'090_210_{ctbin[0]}{ctbin[1]}'
    file_name = efficiency_dir + f'/Eff_Score_{info_string}.npy'

    return {round(e[0], 2): e[1] for e in np.load(file_name).T}


def fill_histo(histo, ctbin, eff, entry, entry_error):
    bin_idx = histo.FindBin((ctbin[0] + ctbin[1]) / 2, round(eff + 0.005, 3))

    histo.SetBinContent(bin_idx, entry)
    histo.SetBinError(bin_idx, entry_error)


def fill_histo_best(histo, ctbin, entry, entry_error):
    bin_idx = histo.FindBin((ctbin[0] + ctbin[1]) / 2)

    histo.SetBinContent(bin_idx, entry)
    histo.SetBinError(bin_idx, entry_error)


def fill_mu(model, ctbin, eff, mass, mass_error):
    fill_histo(MASS_H2[model], ctbin, eff, mass, mass_error)
    

def fill_mu_best(model, ctbin, mass, mass_error):
    fill_histo_best(MASS_BEST[model], ctbin, mass, mass_error)


def fill_shift(model, ctbin, eff, shift, shift_error):
    fill_histo(MASS_SHIFT_H2[model], ctbin, eff, shift, shift_error)
    

def fill_shift_best(model, ctbin, shift, shift_error):
    fill_histo_best(MASS_SHIFT_BEST[model], ctbin, shift, shift_error)


def get_measured_mass(bkg, ctbin, eff):
    bin_idx = MASS_H2[bkg].FindBin((ctbin[0] + ctbin[1]) / 2, round(eff + 0.005, 3))

    mass = MASS_H2[bkg].GetBinContent(bin_idx)
    error = MASS_H2[bkg].GetBinError(bin_idx)

    return mass, error
    
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

###############################################################################
# actual analysis
for split in SPLIT_LIST:
    if SKIP_FITS:
        continue

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
            if eff != eff_best:
                continue

            # define global RooFit objects
            mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2.960, 3.040, 'GeV/c^{2}')
            mass.setVal(MC_MASS)
            delta_mass = ROOT.RooRealVar('delta_m', '#Delta m', -0.0005, 0.0005, 'GeV/c^{2}')
            shift_mass = ROOT.RooAddition('shift_m', "m + #Delta m", ROOT.RooArgList(mass, delta_mass))
               
            # get the data slice as a RooDataSet
            tsd = score_dict[eff]
            roo_data_slice = hau.ndarray2roo(np.array(data_slice.query('score>@tsd')['m'].values, dtype=np.float64), mass)

            # mc slice for the kde
            mc_array = np.array(mc_slice.query('score>@tsd')['m'].values, dtype=np.float64)
            np.random.shuffle(mc_array)
            roo_mc_slice = hau.ndarray2roo(mc_array if len(mc_array) < 5e4 else mc_array[:50000], mass)

            # kde for the signal component
            signal = ROOT.RooKeysPdf('signal', 'signal', shift_mass, mass, roo_mc_slice, ROOT.RooKeysPdf.NoMirror, 2.)

            # mu = ROOT.RooRealVar('mu', 'hypertriton mass', 2.989, 2.993, 'GeV/c^{2}')
            # sigma = ROOT.RooRealVar('sigma', 'hypertriton width', 0.0001, 0.004, 'GeV/c^{2}') \

            # a1 = ROOT.RooRealVar('a1', 'a1', 0, 5.)
            # a2 = ROOT.RooRealVar('a2', 'a2', 0, 10.)
            # n1 = ROOT.RooRealVar('n1', 'n1', 1, 10.)
            # n2 = ROOT.RooRealVar('n2', 'n2', 1, 10.)

            # signal_dscb = ROOT.RooDSCBShape('cb', 'cb', mass, mu, sigma, a1, n1, a2, n2)

            # fit the kde to the MC for systematic estimate
            fit_results_mc = signal.fitTo(roo_mc_slice, ROOT.RooFit.Range(2.960, 3.040), ROOT.RooFit.NumCPU(32), ROOT.RooFit.Save())

            mass_shift = delta_mass.getVal()
            mass_shift_err = delta_mass.getError()

            # fit_results_mc_dscb = signal_dscb.fitTo(roo_mc_slice, ROOT.RooFit.Range(2.960, 3.040), ROOT.RooFit.NumCPU(32), ROOT.RooFit.Save())

            # sigma.setConstant(ROOT.kTRUE)
            # a1.setConstant(ROOT.kTRUE)
            # a2.setConstant(ROOT.kTRUE)
            # n1.setConstant(ROOT.kTRUE)
            # n2.setConstant(ROOT.kTRUE)
            
            frame = mass.frame(80)
            frame.SetName(f'mc_eff{eff:.2f}_{model}')

            roo_mc_slice.plotOn(frame, ROOT.RooFit.Name('MC'))
            signal.plotOn(frame, ROOT.RooFit.Name('signal pdf'), ROOT.RooFit.LineColor(ROOT.kBlue))

            # add info to plot
            pinfo = ROOT.TPaveText(0.537, 0.474, 0.937, 0.875, 'NDC')
            pinfo.SetBorderSize(0)
            pinfo.SetFillStyle(0)
            pinfo.SetTextAlign(30+3)
            pinfo.SetTextFont(42)

            # pinfo.AddText(f'shift = {shift*1e6:.1f} #pm {shift_error*1e6:.1f} keV')

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
                    background = ROOT.RooPolynomial('bkg', 'pol1 bkg', mass, ROOT.RooArgList(c0))

                if model == 'pol2':
                    background = ROOT.RooPolynomial('bkg', 'pol2 bkg', mass, ROOT.RooArgList(c0, c1))

                if model == 'expo':
                    background = ROOT.RooExponential('bkg', 'expo bkg', mass, slope)

                # define fraction
                n = ROOT.RooRealVar('n', 'n const', 0., 1, 'GeV')

                # define the fit funciton and perform the actual fit
                fit_function = ROOT.RooAddPdf(f'{model}_kde', 'signal + background', ROOT.RooArgList(signal, background), ROOT.RooArgList(n))
                fit_results = fit_function.fitTo(roo_data_slice, ROOT.RooFit.Range(2.960, 3.040), ROOT.RooFit.NumCPU(32), ROOT.RooFit.Save())

                # data_kde = signal.generate(mass, 100000)
                # fit_results_dscb = signal_dscb.fitTo(data_kde, ROOT.RooFit.Range(2.960, 3.040), ROOT.RooFit.NumCPU(32), ROOT.RooFit.Save())

                # shift = (MC_MASS - mu.getVal()) * 1e6
                # shift_err = mu.getError() * 1e6

                frame = mass.frame(80)
                frame.SetName(f'eff{eff:.2f}_{model}')

                roo_data_slice.plotOn(frame, ROOT.RooFit.Name('data'))
                fit_function.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Name('model'))
                fit_function.plotOn(frame, ROOT.RooFit.Components('signal'), ROOT.RooFit.LineColor(ROOT.kRed))
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
                # string_list.append(f'#Delta m = {delta_mass.getVal()*1e6:.1f} #pm {delta_mass.getError()*1e6:.1f} keV')
                # string_list.append(f'shift = {mass_shift*1e6:.1f} #pm {mass_shift_err*1e6:.1f} keV')

                for s in string_list:
                    pinfo.AddText(s)

                frame.addObject(pinfo)
                frame.Write()

                fill_mu(model, ctbin, eff, (MC_MASS-delta_mass.getVal()-mass_shift)*1e3, math.sqrt(delta_mass.getError()**2+mass_shift_err**2)*1e3)
                fill_shift(model, ctbin, eff, mass_shift*1e3, mass_shift_err*1e3)
                if eff == eff_best:
                    fill_mu_best(model, ctbin, (MC_MASS-delta_mass.getVal()-mass_shift) * 1e3, math.sqrt(delta_mass.getError()**2+mass_shift_err**2)*1e3)
                    fill_shift_best(model, ctbin, mass_shift*1e3, mass_shift_err*1e3)


                # fill_mu(model, ctbin, eff, (mu.getVal() + shift)*1e3, math.sqrt(mu.getError()**2+shift_error**2)*1e3)
                # fill_shift(model, ctbin, eff, mass_shift*1e3, mass_shift_err*1e3)
                # if eff == eff_best:
                #     fill_mu_best(model, ctbin, (mu.getVal() + shift)*1e3, math.sqrt(mu.getError()**2+shift_error**2)*1e3)
                #     fill_shift_best(model, ctbin, shift*1e3, shift_error*1e3)

    output_file.cd()
    for model in BKG_MODELS:
        MASS_H2[model].Write()
        MASS_BEST[model].Write()

        MASS_SHIFT_H2[model].Write()
        MASS_SHIFT_BEST[model].Write()

        hpu.mass_plot_makeup(MASS_BEST[model], model, CT_BINS, split)
        

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

        # loop over ctbins
        for ctbin_idx in range(len(CT_BINS)-1):
            # random bkg model
            bkg_index = np.random.randint(0, len(BKG_MODELS))
            bkg_idx_list.append(bkg_index)
            bkg_list.append(BKG_MODELS[bkg_index])

            # randon BDT efficiency in the defined range
            eff = np.random.choice(syst_eff_ranges[ctbin_idx])
            eff_list.append(eff)
            eff_index = get_eff_index(eff)
            eff_idx_list.append(eff_index)

        # convert indexes into hash and if already sampled skip this combination
        combo = ''.join(map(str, bkg_idx_list + eff_idx_list))
        if combo in combinations:
            continue

        # if indexes are good measure B_{Lambda}
        ctbin_idx = 1
        ct_bin_it = iter(zip(CT_BINS[:-1], CT_BINS[1:]))

        mass_list = []

        for model, eff in zip(bkg_list, eff_list):
            ctbin = next(ct_bin_it)

            mass, mass_error = get_measured_mass(model, ctbin, eff)

            tmp_mass.SetBinContent(ctbin_idx, mass)
            tmp_mass.SetBinError(ctbin_idx, mass_error)

            mass_list.append(mass)

            ctbin_idx += 1

        mass, mass_error, chi2red = hau.b_form_histo(tmp_mass)
        blambda = 1115.683 + 1875.61294257 - mass

        if chi2red < 3. and mass_error > 0.04:
            blambda_dist.Fill(blambda)
        combinations.add(combo)

    blambda_dist.Write()

###############################################################################
print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')
