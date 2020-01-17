#!/usr/bin/env python3
import argparse
import collections.abc
import os
import time
import warnings
from array import array

import numpy as np
import pandas as pd
import yaml

import analysis_utils as au
import generalized_analysis as ga
import xgboost as xgb
from generalized_analysis import GeneralizedAnalysis
from ROOT import TFile, gROOT


gROOT.SetBatch()

parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
args = parser.parse_args()

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# define paths for loading data and storing results
mc_path = os.path.expandvars(params['MC_PATH'])
data_path = os.path.expandvars(params['DATA_PATH'])

results_dir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

# preselections
signal_selection = '{}<=HypCandPt<={}'.format(params['PT_BINS'][0], params['PT_BINS'][-1])
backgound_selection = '{}<=HypCandPt<={}'.format(params['PT_BINS'][0], params['PT_BINS'][-1])

# start timer for performance evaluation
start_time = time.time()

# initialize support dict
score_bdteff_dict = {}
preselection_efficiency = {}
n_hytr = {}

eff_list = np.arange(0.6, 0.99, 0.01)

file_name_invmass = results_dir + '/' + params['FILE_PREFIX'] + '_results.root'
invmass_file = TFile(file_name_invmass, 'read')

file_name = results_dir + '/' + params['FILE_PREFIX'] + '_results_fit.root'
results_file = TFile(file_name, 'recreate')

for cclass in params['CENTRALITY_CLASS']:
    cent_dirname = '{}-{}'.format(cclass[0], cclass[1])
    cent_dir = results_file.mkdir(f'{cent_dirname}')

    h2seleff = invmass_file.Get(f'{cclass[0]}-{cclass[1]}/SelEff')
    h2seleff.SetDirectory(0)

    if params['FIXED_SIGMA_FIT']:
        h3_invmassptct_list = {}
        h2sigma_mc_list = {}

        for eff in eff_list:
            h3_invmassptct_list['{}'.format(eff)] = au.h3_minvptct(
                params['PT_BINS'], params['CT_BINS'], name='SigmaPtCt{}'.format(eff))
            h2sigma_mc_list['{}'.format(eff)] = au.h2_mcsigma(
                params['PT_BINS'], params['CT_BINS'], name='InvMassPtCt{}'.format(eff))

    bkg_models = params['BKG_MODELS'] if 'BKG_MODELS' in params else['Pol2']

    fit_directories = []
    h2raw_counts = []
    h2significance = []
    h2raw_counts_fixeff_dict = []

    for model in bkg_models:
        fit_directories.append(cent_dir.mkdir(model))

        histo_dict = {}

        for fix_eff in eff_list:
            histo_dict[f'eff{fix_eff:.2f}'] = au.h2_rawcounts(params['PT_BINS'], params['CT_BINS'], f'RawCounts{fix_eff:.2f}_{model}')

        h2raw_counts_fixeff_dict.append(histo_dict)

        h2raw_counts.append(au.h2_rawcounts(params['PT_BINS'], params['CT_BINS'], f'RawCounts_{model}'))
        h2significance.append(au.h2_rawcounts(params['PT_BINS'], params['CT_BINS'], f'significance_{model}'))

    for ptbin in zip(params['PT_BINS'][:-1], params['PT_BINS'][1:]):
        ptbin_index = h2seleff.GetXaxis().FindBin(0.5 * (ptbin[0] + ptbin[1]))

        for ctbin in zip(params['CT_BINS'][:-1], params['CT_BINS'][1:]):
            ctbin_index = h2seleff.GetYaxis().FindBin(0.5 * (ctbin[0] + ctbin[1]))
            ct_dirname = 'ct_{}{}'.format(ctbin[0], ctbin[1])

            # key for accessing the correct value of the dict
            key = 'CENT{}_PT{}_CT{}'.format(cclass, ptbin, ctbin)

            print('============================================')
            print('centrality: ', cclass, ' ct: ', ctbin, ' pT: ', ptbin)

            part_time = time.time()

            total_cut = '{}<ct<{} and {}<HypCandPt<{} and {}<centrality<{}'.format(
                        ctbin[0], ctbin[1], ptbin[0], ptbin[1], cclass[0], cclass[1])

            # df_data = analysis.df_data_all.query(total_cut)

            # extract the signal for each bdtscore-eff configuration
            for eff in eff_list:
                k = f'eff{eff:.2f}'
                # obtain the invariant mass dist
                h1_name = f'ct{ctbin[0]}{ctbin[1]}_pT{ptbin[0]}{ptbin[1]}_cen{cclass[0]}{cclass[1]}{k}'
                h1_minv = invmass_file.Get(f'{cent_dirname}/{ct_dirname}/{h1_name}')

                for model, fitdir, h2raw, h2sig, h2raw_dict in zip(
                        bkg_models, fit_directories, h2raw_counts, h2significance, h2raw_counts_fixeff_dict):

                    hyp_yield, err_yield, signif, errsignif, sigma, sigmaErr = au.fitHist(
                        h1_minv, ctbin, ptbin, cclass, fitdir, model=model)

                    h2raw_dict[k].SetBinContent(ptbin_index, ctbin_index, hyp_yield)
                    h2raw_dict[k].SetBinError(ptbin_index, ctbin_index, err_yield)

    # write on file
    cent_dir.cd()
    h2seleff.Write()

    for h2raw, h2sig in zip(h2raw_counts, h2significance):
        h2raw.Write()
        h2sig.Write()

    for dictionary in h2raw_counts_fixeff_dict:
        for th2 in dictionary.values():
            th2.Write()

results_file.Close()
invmass_file.Close()

# print execution time to performance evaluation
print('')
print('--- {:.4f} minutes ---'.format((time.time() - start_time) / 60))
