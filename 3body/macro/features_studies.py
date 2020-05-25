#!/usr/bin/env python3
import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import yaml

import pandas as pd
from analysis_classes import TrainingAnalysis
from ROOT import TFile, gROOT

parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
args = parser.parse_args()
with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

###############################################################################
# define analysis global variables
N_BODY = params['NBODY']
FILE_PREFIX = params['FILE_PREFIX']

CENT_CLASSES = params['CENTRALITY_CLASS']
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']

COLUMNS = params['TRAINING_COLUMNS']

SPLIT_LIST = ['']

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
bkg_path = os.path.expandvars(params['BKG_PATH'])
data_path = os.path.expandvars(params['DATA_PATH'])
fig_dir = os.environ['HYPERML_FIGURES_{}'.format(N_BODY)]

###############################################################################

def plot_distr(sig_df, bkg_df, column=None, figsize=None, bins=50, log=True, **kwds):
    data1 = bkg_df
    data2 = sig_df

    if column is not None:
        data1 = data1[column]
        data2 = data2[column]

    if figsize is None:
        figsize = [20, 15]

    axes = data1.hist(column=column, color='tab:blue', alpha=0.5, bins=bins, figsize=figsize, label='background', density=True, grid=False, log=log)
    axes = axes.flatten()
    axes = axes[:len(column)]
    data2.hist(ax=axes, column=column, color='tab:orange', alpha=0.5, bins=bins, label='signal', density=True, grid=False, log=log)[0].legend()
    for a in axes:
        a.set_ylabel('Counts (arb. units)')

    return plt

for split in SPLIT_LIST:
    ml_analysis = TrainingAnalysis(N_BODY, signal_path, bkg_path, split)

    for cclass in CENT_CLASSES:
        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
            for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                print('\n==================================================')
                print('centrality:', cclass, ' ct:', ctbin, ' pT:', ptbin, split)

                info_string = f'{cclass[0]}{cclass[1]}_{ptbin[0]}{ptbin[1]}_{ctbin[0]}{ctbin[1]}{split}'

                data_range = f'{ctbin[0]}<ct<{ctbin[1]} and {ptbin[0]}<pt<{ptbin[1]}'

                sig = ml_analysis.df_signal.query(data_range)
                bkg = ml_analysis.df_bkg.query(data_range)

                print('\nNumber of signal candidates: {}'.format(len(sig)))
                print('Number of background candidates: {}\n'.format(len(bkg)))

                vars_to_draw =['cosPA','pt', 'tpcClus_de','tpcClus_pr','tpcClus_pi','tpcNsig_de','tpcNsig_pr','tpcNsig_pi','tofNsig_de','tofNsig_pr','tofNsig_pi','hasTOF_de','hasTOF_pr','hasTOF_pi','dca_de','dca_pr','dca_pi','dca_de_pr','dca_de_pi','dca_pr_pi','chi2_deuprot','chi2_3prongs','chi2_topology']

                leg_labels = ['background', 'signal']

                plot_distr(sig, bkg, vars_to_draw, bins=100, log=True, figsize=(12, 7))
                plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)

                del sig, bkg

                fig_name = f'{fig_dir}/feature_{info_string}.pdf'
                plt.savefig(fig_name)

    del ml_analysis
