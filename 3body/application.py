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
parser.add_argument('-s', '--significance', help='Run the significance optimisation studies', action='store_true')
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

application_columns = ['score', 'InvMass', 'ct', 'HypCandPt', 'centrality']

analysis = GeneralizedAnalysis(3, mc_path, data_path, signal_selection, backgound_selection,
                               cent_class=params['CENTRALITY_CLASS'])


file_name = results_dir + '/' + params['FILE_PREFIX'] + '_results.root'
results_file = TFile(file_name, 'recreate')

for cclass in params['CENTRALITY_CLASS']:
    cent_dir = results_file.mkdir('{}-{}'.format(cclass[0], cclass[1]))

    # create the histos for storing analysis stuff
    h2BDTeff = au.h2_bdteff(params['PT_BINS'], params['CT_BINS'])
    h2seleff = au.h2_seleff(params['PT_BINS'], params['CT_BINS'])

    h2raw_counts = []
    h2raw_counts_fixeff_dict = []

    for ptbin in zip(params['PT_BINS'][:-1], params['PT_BINS'][1:]):
        ptbin_index = h2BDTeff.GetXaxis().FindBin(0.5 * (ptbin[0] + ptbin[1]))

        for ctbin in zip(params['CT_BINS'][:-1], params['CT_BINS'][1:]):
            ctbin_index = h2BDTeff.GetYaxis().FindBin(0.5 * (ctbin[0] + ctbin[1]))

            ct_dir = cent_dir.mkdir('ct_{}{}'.format(ctbin[0], ctbin[1]))
            ct_dir.cd()

            # key for accessing the correct value of the dict
            key = f'CENT{cclass}_PT{ptbin}_CT{ctbin}'

            print('============================================')
            print('centrality: ', cclass, ' ct: ', ctbin, ' pT: ', ptbin)

            part_time = time.time()

            score_bdteff_dict[key] = {}
            preselection_efficiency[key] = analysis.preselection_efficiency(
                ct_range=ctbin, pt_range=ptbin, cent_class=cclass)

            # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
            data = analysis.prep_dataframe(params['TRAINING_COLUMNS'], cclass, ct_range=ctbin, pt_range=ptbin)

            model = analysis.load_model(ct_range=ctbin, cent_class=cclass, pt_range=ptbin)

            if args.significance:
                score_cut, bdt_efficiency = analysis.sig_scan(
                    data[2: 4],
                    model, params['TRAINING_COLUMNS'], ct_range=ctbin, pt_range=ptbin, cent_class=cclass,
                    custom=params['MAX_SIGXEFF'])

                score_bdteff_dict[key]['sig_scan'] = [float(score_cut), float(bdt_efficiency)]

                h2BDTeff.SetBinContent(ptbin_index, ctbin_index, score_bdteff_dict[key]['sig_scan'][1])
            h2seleff.SetBinContent(ptbin_index, ctbin_index, preselection_efficiency[key])

            # compute and store score cut for fixed efficiencies, if required
            score_eff = analysis.score_from_efficiency(
                model, data[2: 4],
                params['BDT_EFFICIENCY'],
                params['TRAINING_COLUMNS'],
                ct_range=ctbin, pt_range=ptbin, cent_class=cclass)

            for se in score_eff:
                score_bdteff_dict[key][f'eff{se[1]:.2f}'] = [float(se[0]), float(se[1])]

            total_cut = '{}<ct<{} and {}<HypCandPt<{} and {}<centrality<{}'.format(
                        ctbin[0], ctbin[1], ptbin[0], ptbin[1], cclass[0], cclass[1])
            df_data = analysis.df_data_all.query(total_cut)

            # extract the signal for each bdtscore-eff configuration
            for k, se in score_bdteff_dict[key].items():
                # obtain the selected invariant mass dist
                mass_bins = 40 if ctbin[1] < 16 else 36

                mass_array = np.array(df_data.query('score >@se[0]')['InvMass'].values, dtype=np.float64)
                counts, _ = np.histogram(mass_array, bins=mass_bins, range=[2.96, 3.05])

                h1_minv = au.h1_invmass(counts, ctbin, ptbin, cclass, bins=mass_bins, name=k)
                h1_minv.Write()

    # write on file
    cent_dir.cd()
    h2BDTeff.Write()
    h2seleff.Write()

results_file.Close()

# print execution time to performance evaluation
print('')
print('--- {:.4f} minutes ---'.format((time.time() - start_time) / 60))
