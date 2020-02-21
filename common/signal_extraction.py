#!/usr/bin/env python3
import argparse
# import collections.abc
import os
import time
import warnings

import numpy as np
import yaml

import hyp_analysis_utils as hau
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
# define dictionaries for storing row counts and significance
h2_rawcounts_dict = {}
significance_dict = {}

###############################################################################
# start the actual signal extraction
for split in SPLIT_LIST:
    for cclass in CENT_CLASSES:
        cent_dir_name = f'{cclass[0]}-{cclass[1]}{split}'
        cent_dir = output_file.mkdir(cent_dir_name)
        cent_dir.cd()

        h2_eff = input_file.Get(f'{cclass[0]}-{cclass[1]}/PreselEff')

        for lab in LABELS:
            h2_rawcounts_dict[lab] = hau.h2_rawcounts(PT_BINS, CT_BINS, name=f'RawCounts{lab}')

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

                for bkgmodel in BKG_MODELS:
                    # create dirs for models
                    fit_dir = output_subdir.mkdir(bkgmodel)
                    fit_dir.cd()

                    # loop over all the histo in the dir
                    for key in input_subdir.GetListOfKeys():
                        hist = TH1D(key.ReadObj())
                        hist.SetDirectory(0)

                        rawcounts, err_rawcounts, significance, err_significance, sigma, err_sigma = hau.fit_hist(
                            hist, cclass, ptbin, ctbin, model=bkgmodel, mode=3)

                        dict_key = f'{key.GetName()[-4:]}_{bkgmodel}'

                        h2_rawcounts_dict[dict_key].SetBinContent(ptbin_index, ctbin_index, rawcounts)
                        h2_rawcounts_dict[dict_key].SetBinError(ptbin_index, ctbin_index, err_rawcounts)

                        # significance_dict[dict_key].SetBinContent(ptbin_index, ctbin_index, significance)
                        # significance_dict[dict_key]SetBinError(ptbin_index, ctbin_index, err_significance)

        cent_dir.cd()
        h2_eff.Write()

        for lab in LABELS:
            h2_rawcounts_dict[lab].Write()

output_file.Close()


# for cclass in params['CENTRALITY_CLASS']:
#     cent_dirname = '{}-{}'.format(cclass[0], cclass[1])
#     cent_dir = results_file.mkdir(f'{cent_dirname}')

#     h2seleff = invmass_file.Get(f'{cclass[0]}-{cclass[1]}/SelEff')
#     h2seleff.SetDirectory(0)

#     if params['FIXED_SIGMA_FIT']:
#         h3_invmassptct_list = {}
#         h2sigma_mc_list = {}

#         for eff in eff_list:
#             h3_invmassptct_list['{}'.format(eff)] = au.h3_minvptct(
#                 params['PT_BINS'], params['CT_BINS'], name='SigmaPtCt{}'.format(eff))
#             h2sigma_mc_list['{}'.format(eff)] = au.h2_mcsigma(
#                 params['PT_BINS'], params['CT_BINS'], name='InvMassPtCt{}'.format(eff))

#     bkg_models = params['BKG_MODELS'] if 'BKG_MODELS' in params else['Pol2']

#     fit_directories = []
#     h2raw_counts = []
#     h2significance = []
#     h2raw_counts_fixeff_dict = []

#     for model in bkg_models:
#         fit_directories.append(cent_dir.mkdir(model))

#         histo_dict = {}

#         for fix_eff in eff_list:
#             histo_dict[f'eff{fix_eff:.2f}'] = au.h2_rawcounts(params['PT_BINS'], params['CT_BINS'], f'RawCounts{fix_eff:.2f}_{model}')

#         h2raw_counts_fixeff_dict.append(histo_dict)

#         h2raw_counts.append(au.h2_rawcounts(params['PT_BINS'], params['CT_BINS'], f'RawCounts_{model}'))
#         h2significance.append(au.h2_rawcounts(params['PT_BINS'], params['CT_BINS'], f'significance_{model}'))

#     for ptbin in zip(params['PT_BINS'][:-1], params['PT_BINS'][1:]):
#         ptbin_index = h2seleff.GetXaxis().FindBin(0.5 * (ptbin[0] + ptbin[1]))

#         for ctbin in zip(params['CT_BINS'][:-1], params['CT_BINS'][1:]):
#             ctbin_index = h2seleff.GetYaxis().FindBin(0.5 * (ctbin[0] + ctbin[1]))
#             ct_dirname = 'ct_{}{}'.format(ctbin[0], ctbin[1])

#             # key for accessing the correct value of the dict
#             key = 'CENT{}_PT{}_CT{}'.format(cclass, ptbin, ctbin)

#             print('============================================')
#             print('centrality: ', cclass, ' ct: ', ctbin, ' pT: ', ptbin)

#             part_time = time.time()

#             total_cut = '{}<ct<{} and {}<HypCandPt<{} and {}<centrality<{}'.format(
#                         ctbin[0], ctbin[1], ptbin[0], ptbin[1], cclass[0], cclass[1])

#             # df_data = analysis.df_data_all.query(total_cut)

#             # extract the signal for each bdtscore-eff configuration
#             for eff in eff_list:
#                 k = f'eff{eff:.2f}'
#                 # obtain the invariant mass dist
#                 h1_name = f'ct{ctbin[0]}{ctbin[1]}_pT{ptbin[0]}{ptbin[1]}_cen{cclass[0]}{cclass[1]}{k}'
#                 h1_minv = invmass_file.Get(f'{cent_dirname}/{ct_dirname}/{h1_name}')

#                 for model, fitdir, h2raw, h2sig, h2raw_dict in zip(
#                         bkg_models, fit_directories, h2raw_counts, h2significance, h2raw_counts_fixeff_dict):

#                     hyp_yield, err_yield, signif, errsignif, sigma, sigmaErr = au.fit_hist(
#                         h1_minv, ctbin, ptbin, cclass, fitdir, model=model, mode=3)

# h2raw_dict[k].SetBinContent(ptbin_index, ctbin_index, hyp_yield)
# h2raw_dict[k].SetBinError(ptbin_index, ctbin_index, err_yield)

#     # write on file
#     cent_dir.cd()
#     h2seleff.Write()

#     for h2raw, h2sig in zip(h2raw_counts, h2significance):
#         h2raw.Write()
#         h2sig.Write()

#     for dictionary in h2raw_counts_fixeff_dict:
#         for th2 in dictionary.values():
#             th2.Write()

# results_file.Close()
# invmass_file.Close()

# # print execution time to performance evaluation
# print('')
# print('--- {:.4f} minutes ---'.format((time.time() - start_time) / 60))
