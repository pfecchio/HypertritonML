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
from ROOT import RooDSCBShape

ROOT.gROOT.SetBatch()

np.random.seed(42)

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
file_name = results_dir + f'/{FILE_PREFIX}_blambda{suffix}.root'
output_file = ROOT.TFile(file_name, 'recreate')
###############################################################################

file_name = results_dir + f'/{FILE_PREFIX}_signal_extraction{suffix}.root'
input_file = ROOT.TFile(file_name)

##load dbshape in any case for systematics
if not DBSHAPE:
    file_name_dbshape = results_dir + f'/{FILE_PREFIX}_signal_extraction_dscb.root'
    input_file_dbshape = ROOT.TFile(file_name_dbshape)
    

###############################################################################
start_time = time.time()
###############################################################################

###############################################################################
# define support globals 
MASS_H2 = {}
MASS_SHIFT_H2 = {}
RECO_SHIFT_H2 = {}


MASS_BEST = {}
MASS_SHIFT_BEST = {}
RECO_SHIFT_BEST = {}

# support for systematics
MASS_H2_DBSHAPE = {}
RECO_SHIFT_H2_DBSHAPE = {}


MC_MASS = 2.99131

# prepare histograms for the analysis
for model in BKG_MODELS:
        MASS_H2[model] = input_file.Get(f'mass_{model}')
        MASS_BEST[model] = MASS_H2[model].ProjectionX(f'mass_best_{model}')
        if not DBSHAPE:
            MASS_SHIFT_H2[model] = input_file.Get(f'mass_shift_{model}')
            MASS_SHIFT_H2[model].SetDirectory(0)
            MASS_SHIFT_BEST[model] = MASS_BEST[model].Clone("mass_shift_best")
            MASS_H2_DBSHAPE[model] = input_file_dbshape.Get(f'mass_{model}')
            MASS_H2_DBSHAPE[model].SetDirectory(0)
            RECO_SHIFT_H2_DBSHAPE[model] = input_file_dbshape.Get(f'reco_shift_{model}')
            RECO_SHIFT_H2_DBSHAPE[model].SetDirectory(0)


        else:
            RECO_SHIFT_H2[model] = input_file.Get(f'reco_shift_{model}')
            RECO_SHIFT_BEST[model] = MASS_BEST[model].Clone("reco_shift_best") 

# helper methods
def get_eff_index(eff):
    idx = (eff - EFF_MIN + EFF_STEP) * 100
    if isinstance(eff, np.ndarray):
        return idx.astype(int)
    return int(idx)

def fill_histo_best(histo, ctbin, entry, entry_error):
    bin_idx = histo.FindBin((ctbin[0] + ctbin[1]) / 2)
    histo.SetBinContent(bin_idx, entry)
    histo.SetBinError(bin_idx, entry_error)


def get_measured_h2(h2, bkg, ctbin, eff):
    bin_idx = h2[bkg].FindBin((ctbin[0] + ctbin[1]) / 2, round(eff + 0.005, 3))
    var = h2[bkg].GetBinContent(bin_idx)
    error = h2[bkg].GetBinError(bin_idx)
    return var, error



# def get_h1_frame()
###############################################################################

# significance-scan/fixed efficiencies switch
if not SIGNIFICANCE_SCAN:
    eff_best_array = np.full(len(CT_BINS) - 1, FIX_EFF)
else:
    eff_best_array = [round(sigscan_dict[f'ct{ctbin[0]}{ctbin[1]}pt210'][0], 2) for ctbin in zip(CT_BINS[:-1], CT_BINS[1:])]

output_file.mkdir('fits')
output_file.cd('fits')

for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
    eff = round(sigscan_dict[f'ct{ctbin[0]}{ctbin[1]}pt210'][0], 2)
    print("BDT eff: ", eff)
    for model in BKG_MODELS:
        mass, mass_error = get_measured_h2(MASS_H2, model, ctbin, eff)
        if not DBSHAPE:
            fill_histo_best(MASS_BEST[model], ctbin, mass, mass_error)
            mass_shift, mass_shift_error = get_measured_h2(MASS_SHIFT_H2, model, ctbin, eff)
            fill_histo_best(MASS_SHIFT_BEST[model], ctbin, mass_shift, mass_shift_error)
        else:
            reco_shift, reco_shift_error = get_measured_h2(RECO_SHIFT_H2, model, ctbin, eff)
            fill_histo_best(RECO_SHIFT_BEST[model], ctbin, reco_shift, reco_shift_error)
            fill_histo_best(MASS_BEST[model], ctbin, mass, mass_error)    



output_file.cd()
for model in BKG_MODELS:
    MASS_BEST[model].Write()
    MASS_SHIFT_BEST[model].Write() if not DBSHAPE else RECO_SHIFT_BEST[model].Write()
    if DBSHAPE:
        MASS_BEST[model].Add(RECO_SHIFT_BEST[model]) 
    else:
       MASS_BEST[model].Add(MASS_SHIFT_BEST[model])
  
    hpu.mass_plot_makeup(MASS_BEST[model], model, CT_BINS, "")


# efficiency ranges for sampling the systematics
syst_eff_ranges = np.asarray([list(range(int(x * 100) - 10, int(x * 100) + 11)) for x in eff_best_array]) / 100

if SYSTEMATICS:
    # systematics histos
    blambda_dist = ROOT.TH1D('syst_blambda', ';B_{#Lambda} MeV ;counts', 100, -0.5, 0.5)

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
            mass, mass_error = get_measured_h2(MASS_H2, model, ctbin, eff)
            if DBSHAPE:
                reco_shift, reco_shift_error = get_measured_h2(RECO_SHIFT_H2, model, ctbin, eff)
                mass += reco_shift
                mass_error = np.sqrt(mass_error**2 + reco_shift_error**2)
            else:
                fit_shift, fit_shift_error = get_measured_h2(MASS_SHIFT_H2, model, ctbin, eff)
                mass += fit_shift
                mass_error = np.sqrt(mass_error**2 + fit_shift_error**2)
                switch_to_db = np.random.randint(2)
                if switch_to_db:
                    mass, mass_error = get_measured_h2(MASS_H2_DBSHAPE, model, ctbin, eff)
                    reco_shift, reco_shift_error = get_measured_h2(RECO_SHIFT_H2_DBSHAPE, model, ctbin, eff)
                    mass += reco_shift
                    mass_error = np.sqrt(mass_error**2 + reco_shift_error**2)


            tmp_mass.SetBinContent(ctbin_idx, mass)
            tmp_mass.SetBinError(ctbin_idx, mass_error)

            mass_list.append(mass)

            ctbin_idx += 1

        mass, mass_error, chi2red = hau.b_form_histo(tmp_mass)
        blambda = 1115.683 + 0.036 + 1875.61294257 - mass

        if chi2red < 2.:
            blambda_dist.Fill(blambda)
            combinations.add(combo)

    blambda_dist.Write()

###############################################################################
print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')
