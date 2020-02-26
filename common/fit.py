#!/usr/bin/env python3
import argparse
# import collections.abc
import os
import time
import warnings

import numpy as np
import yaml

import hyp_analysis_utils as hau

from analysis_classes import load_mcsigma
from ROOT import TH1D, TFile, gROOT

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
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

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)

SIGMA_MC = params['SIGMA_MC']
BKG_MODELS = params['BKG_MODELS']

SPLIT_MODE = args.split

if SPLIT_MODE:
    SPLIT_LIST = ['_matter', '_antimatter']
else:
    SPLIT_LIST = ['']

LABELS = [f'{x:.2f}_{y}' for x in FIX_EFF_ARRAY for y in BKG_MODELS]

###############################################################################
# define paths for loading results
results_dir = os.environ['HYPERML_RESULTS_{}'.format(N_BODY)]

input_file_name = results_dir + f'/{FILE_PREFIX}_results.root'
input_file = TFile(input_file_name, 'read')

output_file_name = results_dir + f'/{FILE_PREFIX}_results_fit.root'
output_file = TFile(output_file_name, 'recreate')

###############################################################################
# define dictionaries for storing raw counts and significance
h2_rawcounts_dict = {}
significance_dict = {}

# if not specified do not use MC sigma
mcsigma = -1

###############################################################################
# start the actual signal extraction
for split in SPLIT_LIST:
    for cclass in CENT_CLASSES:
        cent_dir_name = f'{cclass[0]}-{cclass[1]}{split}'
        cent_dir = output_file.mkdir(cent_dir_name)
        cent_dir.cd()

        h2_eff = input_file.Get(cent_dir_name + '/PreselEff')
        h2_BDT_eff = hau.h2_rawcounts(PT_BINS, CT_BINS, name = "BDTeff")

        for lab in LABELS:
            h2_rawcounts_dict[lab] = hau.h2_rawcounts(PT_BINS, CT_BINS, suffix=lab)
            significance_dict[lab] = hau.h2_significance(PT_BINS, CT_BINS, suffix=lab)

        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
            ptbin_index = hau.get_ptbin_index(h2_eff, ptbin)

            for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                ctbin_index = hau.get_ctbin_index(h2_eff, ctbin)

                # get the dir where the inv mass histo are
                subdir_name = f'ct_{ctbin[0]}{ctbin[1]}' if 'ct' in FILE_PREFIX else f'pt_{ptbin[0]}{ptbin[1]}'
                input_subdir = input_file.Get(f'{cent_dir_name}/{subdir_name}')

                # create the subdir in the output file
                output_subdir = cent_dir.mkdir(subdir_name)
                output_subdir.cd()

                if SIGMA_MC:
                    sigma_dict = load_mcsigma(cclass, ptbin, ctbin, N_BODY, split)

                for bkgmodel in BKG_MODELS:
                    # create dirs for models
                    fit_dir = output_subdir.mkdir(bkgmodel)
                    fit_dir.cd()

                    # loop over all the histo in the dir
                    for key in input_subdir.GetListOfKeys():
                        keff = key.GetName()[-4:]
                       
                        hist = TH1D(key.ReadObj())
                        hist.SetDirectory(0)

                        if SIGMA_MC:
                            mcsigma = sigma_dict.item().get(keff)

                        rawcounts, err_rawcounts, significance, err_significance, _, _ = hau.fit_hist(hist, cclass, ptbin, ctbin, model=bkgmodel, fixsigma=mcsigma , mode=N_BODY)

                        dict_key = f'{keff}_{bkgmodel}'

                        h2_rawcounts_dict[dict_key].SetBinContent(ptbin_index, ctbin_index, rawcounts)
                        h2_rawcounts_dict[dict_key].SetBinError(ptbin_index, ctbin_index, err_rawcounts)

                        significance_dict[dict_key].SetBinContent(ptbin_index, ctbin_index, significance)
                        significance_dict[dict_key].SetBinError(ptbin_index, ctbin_index, err_significance)

                        if key == input_subdir.GetListOfKeys()[0]:
                            h2_BDT_eff.SetBinContent(ptbin_index, ctbin_index, float(keff))                           

        cent_dir.cd()
        h2_eff.Write()
        h2_BDT_eff.Write()
        for lab in LABELS:
            h2_rawcounts_dict[lab].Write()

output_file.Close()
