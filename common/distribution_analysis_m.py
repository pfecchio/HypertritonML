#!/usr/bin/env python3

import argparse
import math
import os

import hyp_plot_utils as hpu
import numpy as np
import ROOT
import yaml
from scipy import stats

ROOT.gROOT.SetBatch()

np.random.seed(42)

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('-s', '--scan', help='Use the BDTefficiency selection from the significance scan', action='store_true')
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

LAMBDA_MASS = [1.115683, 0.000006]
DEUTERON_MASS = [1.87561294257, 0.00000057]

SPLIT_LIST = ['_matter','_antimatter'] if args.split else ['']
BKG_MODELS = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']

CENT_CLASS = params['CENTRALITY_CLASS'][0]
PT_BINS = params['PT_BINS']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)

FIX_EFF = 0.40

SYSTEMATICS_COUNTS = 10000
###############################################################################

###############################################################################
# input/output files
results_dir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

# output file
file_name = results_dir + f'/{FILE_PREFIX}_mass.root'
output_file = ROOT.TFile(file_name, 'recreate')

# mass shift correction file
file_name = results_dir + f'/{FILE_PREFIX}_mass_shift.root'
mass_correction_file = ROOT.TFile(file_name, 'read')

# input file
file_name = results_dir + f'/{FILE_PREFIX}_results_unbinned.root'

results_file = ROOT.TFile(file_name, 'read')

# significance scan output
file_name = results_dir + f'/Efficiencies/{FILE_PREFIX}_sigscan.npy'
sigscan_dict = np.load(file_name, allow_pickle=True).item()
###############################################################################

###############################################################################
# define support globals and methods for getting mass, width and related errors
MASS_H2 = {}
MASS_BEST = {}

WIDTH_H2 = {}
WIDTH_BEST = {}

def get_eff_index(eff):
    idx = (eff - EFF_MIN + EFF_STEP) * 100

    if isinstance(eff, np.ndarray):
        return idx.astype(int)

    return int(idx)

def get_measured_mass(bkg_model, ptbin, eff):
    eff_idx = get_eff_index(eff)
    mass = MASS_H2[f'{model}'].GetBinContent(ptbin, eff_idx)
    mass_error = MASS_H2[f'{model}'].GetBinError(ptbin, eff_idx)

    return mass, mass_error

def get_measured_width(bkg_model, ptbin, eff):
    eff_idx = get_eff_index(eff)
    width = WIDTH_H2[f'{model}'].GetBinContent(ptbin, eff_idx)
    width_error = WIDTH_H2[f'{model}'].GetBinError(ptbin, eff_idx)

    return width, width_error
###############################################################################

for split in SPLIT_LIST:
    # fit value -> B_{Lambda}
    pol0 = ROOT.TF1('blfunction', '1115.683 + 1875.61294257 - [0]', 0, 10)

    hist_shift_h2 = mass_correction_file.Get('shift_fit' + split)
    tmp_mass = hist_shift_h2.ProjectionX('tmp_mass')

    input_dirname = f'{CENT_CLASS[0]}-{CENT_CLASS[1]}{split}'

    # significance-scan/fixed efficiencies switch
    best_sig = np.round(list(sigscan_dict.values()), 2).T[0] if args.scan else np.full(7, FIX_EFF)

    # efficiency ranges for sampling the systematics
    syst_eff_ranges = (np.hstack([range(int(x * 100) - 10, int(x * 100) + 11) for x in best_sig]) / 100).reshape(-1, 7)
    # and related indexes
    syst_eff_idx = get_eff_index(syst_eff_ranges)
        
    results_file.cd()
    out_dir = output_file.mkdir(input_dirname)

    for model in BKG_MODELS:
        # initialize histos for the measured mass vs pT vs BDT efficiency
        MASS_H2[f'{model}'] = hist_shift_h2.Clone(f'mass_{model}{split}')
        MASS_H2[f'{model}'].Reset()

        MASS_BEST[f'{model}'] = hist_shift_h2.ProjectionX(f'mass_best_{model}')
        MASS_BEST[f'{model}'].Reset()

        # initialize histos for the measured sigma vs pT vs BDT efficiency
        WIDTH_H2[f'{model}'] = hist_shift_h2.Clone(f'width_{model}{split}')
        WIDTH_H2[f'{model}'].Reset()

        WIDTH_BEST[f'{model}'] = hist_shift_h2.ProjectionX(f'width_best_{model}')
        WIDTH_BEST[f'{model}'].Reset()
                
        out_dir.cd()
        for ptbin_idx in range(1, len(PT_BINS)):
            effbin_idx = 0

            for eff in EFF_ARRAY:
                # get the results of the unbinned fit
                sub_dir_name = f'pT{PT_BINS[ptbin_idx-1]}{PT_BINS[ptbin_idx]}_eff{eff:.2f}{split}'
                roo_hyp_mass = results_file.Get(sub_dir_name + f'/hyp_mass_model{model}')
                roo_width = results_file.Get(sub_dir_name + f'/width_model{model}')

                mass = roo_hyp_mass.getVal()
                mass_err = roo_hyp_mass.getError()

                width = roo_width.getVal()
                width_err = roo_width.getError()

                # fill the measured mass histogram
                effbin_idx += 1
                MASS_H2[f'{model}'].SetBinContent(ptbin_idx, effbin_idx, mass)
                MASS_H2[f'{model}'].SetBinError(ptbin_idx, effbin_idx, mass_err)

                # fill the measured width histogram
                WIDTH_H2[f'{model}'].SetBinContent(ptbin_idx, effbin_idx, width)
                WIDTH_H2[f'{model}'].SetBinError(ptbin_idx, effbin_idx, width_err)

        # GeV to MeV conversion and shift correction for the mass
        WIDTH_H2[f'{model}'].Scale(1000.)
        MASS_H2[f'{model}'].Scale(1000.)
        MASS_H2[f'{model}'].Add(hist_shift_h2, -1)

        out_dir.cd()
        MASS_H2[f'{model}'].Write()
        WIDTH_H2[f'{model}'].Write()

        for ptbin_idx in range(1, len(PT_BINS)):
            # get the mass measurements
            m, m_err = get_measured_mass(model, ptbin_idx, best_sig[ptbin_idx-1])
            MASS_BEST[f'{model}'].SetBinContent(ptbin_idx, m)
            MASS_BEST[f'{model}'].SetBinError(ptbin_idx, m_err)

            # get the width measurements
            w, w_err = get_measured_width(model, ptbin_idx, best_sig[ptbin_idx-1])
            WIDTH_BEST[f'{model}'].SetBinContent(ptbin_idx, w)
            WIDTH_BEST[f'{model}'].SetBinError(ptbin_idx, w_err)

        # just histo makeup and canvas generation
        hpu.mass_plot_makeup(MASS_BEST[f'{model}'], pol0, model, PT_BINS, split)
        hpu.sigma_plot_makeup(WIDTH_BEST[f'{model}'], model, PT_BINS, split)

    # systematics histos
    blambda_dist = ROOT.TH1D('syst_blambda', ';B_{#Lambda} MeV ;counts', 150, -0.15, 0.15)
    blambda_prob = ROOT.TH1D('prob_blambda', ';prob. ;counts', 200, 0, 1)

    combinations = set()
    sample_counts = 0   # good fits
    iterations = 0  # total fits

    # stop with SYSTEMATICS_COUNTS good B_{Lambda} fits
    while sample_counts < SYSTEMATICS_COUNTS:
        tmp_mass.Reset()

        iterations += 1

        bkg_idx_list = []
        eff_idx_list = []

        # loop over ptbins
        for ptbin_idx in range(1, len(PT_BINS)):
            # random bkg model
            bkg_index = np.random.randint(0, 2)
            bkg_idx_list.append(bkg_index)
            # randon BDT efficiency in the defined range
            eff_index = np.random.choice(syst_eff_idx[ptbin_idx - 1])
            eff_idx_list.append(eff_index)

        # convert indexes into hash and if already sampled skip this combination
        combo = ''.join(map(str, bkg_idx_list + eff_idx_list))
        if combo in combinations:
            continue

        # if indexes are good measure B_{Lambda}
        ptbin_idx = 1
        for bkg_idx, eff_idx in zip(bkg_idx_list, eff_idx_list):
            model = BKG_MODELS[bkg_index]

            mass = MASS_H2[f'{model}'].GetBinContent(ptbin_idx, int(eff_idx))
            mass_err = MASS_H2[f'{model}'].GetBinError(ptbin_idx, int(eff_idx))

            tmp_mass.SetBinContent(ptbin_idx, mass)
            tmp_mass.SetBinError(ptbin_idx, mass_err)

            ptbin_idx += 1

        tmp_mass.Fit(pol0)

        # if B_{Lambda} fit is good use it for systematics
        if pol0.GetChisquare() > 3 * pol0.GetNDF():
            continue

        blambda_dist.Fill(pol0.GetParameter(0))
        blambda_prob.Fill(pol0.GetProb())

        combinations.add(combo)
        sample_counts += 1

    blambda_dist.Write()
    blambda_prob.Write()

    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'\nGood iterations / Total iterations -> {SYSTEMATICS_COUNTS/iterations:.4f}')
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
