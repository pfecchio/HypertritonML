#!/usr/bin/env python3
import argparse
import collections.abc
import os
import time
import warnings

import analysis_utils as au
import generalized_analysis as ga
import pandas as pd
import xgboost as xgb
from generalized_analysis import GeneralizedAnalysis
import yaml

from array import array
import numpy as np

from ROOT import TFile, TF1, TH2D, TH1D, TCanvas, TPaveText, gStyle, gROOT

gROOT.SetBatch()

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='Do the training', action='store_true')
parser.add_argument(
    '--test', help='Just test the functionalities (training with reduced number of candidates)', action='store_true')
parser.add_argument('-o', '--optimize', help='Run the optimization', action='store_true')
parser.add_argument('-s', '--significance', help='Run the significance optimisation studies', action='store_true')
parser.add_argument('-a', '--anti', help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('-m', '--matter', help='Run with matter and anti-matter splitted', action='store_true')
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

# initialize support dict
score_bdteff_dict = {}
preselection_efficiency = {}
n_hytr = {}

# load saved score-BDTeff dict or open a file for saving the new ones
score_bdteff_name = results_dir + '/{}_score_bdteff.yaml'.format(params['FILE_PREFIX'])
if params['LOAD_SCORE_EFF']:
    with open(score_bdteff_name, 'r') as stream:
        try:
            score_bdteff_dict = yaml.full_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

# define data selection
if args.matter:
    signal_selection = '{}<=HypCandPt<={} and ArmenterosAlpha > 0'.format(params['PT_BINS'][0], params['PT_BINS'][-1])
    backgound_selection = '(InvMass<2.98 or InvMass>3.005) and {}<=HypCandPt<={} and ArmenterosAlpha > 0'.format(
        params['PT_BINS'][0], params['PT_BINS'][-1])

if args.anti:
    signal_selection = '{}<=HypCandPt<={} and ArmenterosAlpha < 0'.format(params['PT_BINS'][0], params['PT_BINS'][-1])
    backgound_selection = '(InvMass<2.98 or InvMass>3.005) and {}<=HypCandPt<={} and ArmenterosAlpha < 0'.format(
        params['PT_BINS'][0], params['PT_BINS'][-1])

if not args.matter and not args.anti:
    signal_selection = '{}<=HypCandPt<={}'.format(params['PT_BINS'][0], params['PT_BINS'][-1])
    backgound_selection = '(InvMass<2.98 or InvMass>3.005) and {}<=HypCandPt<={}'.format(
        params['PT_BINS'][0], params['PT_BINS'][-1])

split = 0

if args.anti:
    split = 'a'
if args.matter:
    split = 'm'

bkgReservedFraction = params['DEDICATED_BACKGROUND'] if 'DEDICATED_BACKGROUND' in params else 0

# initilize the analysis object
analysis = GeneralizedAnalysis(params['NBODY'], mc_path, data_path,
                               signal_selection, backgound_selection,
                               cent_class=params['CENTRALITY_CLASS'], split=split,
                               dedicated_background=bkgReservedFraction, training_columns=params['TRAINING_COLUMNS'])

# start timer for performance evaluation
start_time = time.time()

# params for config the analysis
optimisation_params = params['HYPERPARAMS'] if params['OPTIMIZATION_STRATEGY'] == 'gs' else params['HYPERPARAMS_RANGE']
optimisation_strategy = 'gs' if params['OPTIMIZATION_STRATEGY'] == 'gs' else 'bayes'


file_name = results_dir + '/' + params['FILE_PREFIX'] + '_results.root'
results_file = TFile(file_name, 'recreate')


for cclass in params['CENTRALITY_CLASS']:
    cent_dir = results_file.mkdir('{}-{}'.format(cclass[0], cclass[1]))

    # create the histos for storing analysis stuff
    h2BDTeff = TH2D('BDTeff', ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);BDT efficiency', len(params['PT_BINS'])-1, np.array(
        params['PT_BINS'], 'double'), len(params['CT_BINS'])-1, np.array(params['CT_BINS'], 'double'))
    h2SelEff = TH2D('SelEff', ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Preselection efficiency', len(params['PT_BINS'])-1, np.array(
        params['PT_BINS'], 'double'), len(params['CT_BINS'])-1, np.array(params['CT_BINS'], 'double'))

    bkgModels = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']
    fitDirectories = []
    h2RawCounts = []
    h2RawCountsFixEffDict = []
    for model in bkgModels:
        fitDirectories.append(cent_dir.mkdir(model))
        if params['BDT_EFF_CUTS']:
            myDict = {}

            for fix_eff in params['BDT_EFFICIENCY']:
                myDict['eff{}'.format(fix_eff)] = TH2D('RawCounts{}_{}'.format(fix_eff, model), ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Raw counts', len(
                    params['PT_BINS'])-1, np.array(params['PT_BINS'], 'double'), len(params['CT_BINS']) - 1, np.array(params['CT_BINS'], 'double'))

            h2RawCountsFixEffDict.append(myDict)

        h2RawCounts.append(TH2D('RawCounts_{}'.format(model), ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Raw counts',
                                len(params['PT_BINS'])-1, np.array(params['PT_BINS'], 'double'), len(params['CT_BINS']) - 1,
                                np.array(params['CT_BINS'], 'double')))

    for ptbin in zip(params['PT_BINS'][:-1], params['PT_BINS'][1:]):
        ptbin_index = h2BDTeff.GetXaxis().FindBin(0.5 * (ptbin[0] + ptbin[1]))

        for ctbin in zip(params['CT_BINS'][:-1], params['CT_BINS'][1:]):
            ctbin_index = h2BDTeff.GetYaxis().FindBin(0.5 * (ctbin[0] + ctbin[1]))

            # key for accessing the correct value of the dict
            key = 'CENT{}_PT{}_CT{}'.format(cclass, ptbin, ctbin)

            print('============================================')
            print('centrality: ', cclass, ' ct: ', ctbin, ' pT: ', ptbin)
            part_time = time.time()

            score_bdteff_dict[key] = {}
            preselection_efficiency[key] = analysis.preselection_efficiency(
                ct_range=ctbin, pt_range=ptbin, cent_class=cclass)

            # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
            data = analysis.prepare_dataframe(
                params['TRAINING_COLUMNS'],
                cclass, ct_range=ctbin, pt_range=ptbin, test=args.test, sig_nocent=True)

            # train the models if required or load trained models
            if args.train:
                # train and test the model with some performance plots
                model = analysis.train_test_model(
                    data, params['TRAINING_COLUMNS'], params['XGBOOST_PARAMS'],
                    hyperparams=optimisation_params, ct_range=ctbin,
                    cent_class=cclass, pt_range=ptbin, optimize=args.optimize,
                    optimize_mode=optimisation_strategy, num_rounds=500, es_rounds=20)
                print('--- model trained in {:.4f} minutes ---\n'.format((time.time() - part_time) / 60))
                analysis.save_model(model, ct_range=ctbin, cent_class=cclass, pt_range=ptbin)
            else:
                model = analysis.load_model(ct_range=ctbin, cent_class=cclass, pt_range=ptbin)

            # significance scan if required, store best score cut and related bdt eff
            if args.significance and not params['LOAD_SCORE_EFF']:
                score_cut, bdt_efficiency = analysis.significance_scan(
                    data[2: 4],
                    model, params['TRAINING_COLUMNS'],
                    ct_range=ctbin, pt_range=ptbin, cent_class=cclass, custom=params['MAX_SIGXEFF'],
                    n_points=100)

                score_bdteff_dict[key]['sig_scan'] = [float(score_cut), float(bdt_efficiency)]

                h2BDTeff.SetBinContent(ptbin_index, ctbin_index, score_bdteff_dict[key]['sig_scan'][1])
            h2SelEff.SetBinContent(ptbin_index, ctbin_index, preselection_efficiency[key])

            # compute and store score cut for fixed efficiencies, if required
            if params['BDT_EFF_CUTS'] and not params['LOAD_SCORE_EFF']:
                score_eff = analysis.score_from_efficiency(
                    model, data[2: 4],
                    params['BDT_EFFICIENCY'],
                    params['TRAINING_COLUMNS'],
                    ct_range=ctbin, pt_range=ptbin, cent_class=cclass)

                for se in score_eff:
                    score_bdteff_dict[key]['eff{}'.format(se[1])] = [float(se[0]), float(se[1])]

            # prediction on the test set for systematics only if required
            # if params['SYST_UNCERTANTIES']:
            #     dtest = xgb.DMatrix(data=(data[2][params['TRAINING_COLUMNS']]))
            #     y_pred = model.predict(dtest, output_margin=True)

            #     data[2].eval('Score = @y_pred', inplace=True)
            #     data[2].eval('y = @data[3]', inplace=True)

            total_cut = '{}<ct<{} and {}<HypCandPt<{} and {}<centrality<{}'.format(
                ctbin[0], ctbin[1], ptbin[0], ptbin[1], cclass[0], cclass[1])

            dfDataF = analysis.df_data_all.query(total_cut)
            data = xgb.DMatrix(data=(analysis.df_data_all.query(total_cut)[params['TRAINING_COLUMNS']]))

            y_pred = model.predict(data, output_margin=True)
            dfDataF.eval('Score = @y_pred', inplace=True)

            # extract the signal for each bdtscore-eff configuration
            for k, se in score_bdteff_dict[key].items():
                # systematics stuff
                # if params['SYST_UNCERTANTIES']:
                #     bdt_efficiency_vars = []
                #     score_cut_vars = []
                #     for shift in params['CUT_SHIFT']:
                #         bdt_efficiency_vars.append(analysis.bdt_efficiency(data[2], score_cut + shift))
                #         score_cut_vars.append(score_cut + shift)

                # obtain the selected invariant mass dist
                mass_bins = 40 if ctbin[1] < 16 else 30

                for model, fitDir, h2Raw, h2RawDict in zip(
                        bkgModels, fitDirectories, h2RawCounts, h2RawCountsFixEffDict):
                    counts, bins = np.histogram(
                        dfDataF.query('Score >@se[0]')['InvMass'],
                        bins=mass_bins, range=[2.96, 3.05])

                    hypYield, eYield = au.fit(counts, ctbin, ptbin, cclass, fitDir, name=k, bins=mass_bins, model=model)

                    if k is 'sig_scan':
                        h2Raw.SetBinContent(ptbin_index, ctbin_index, hypYield)
                        h2Raw.SetBinError(ptbin_index, ctbin_index, eYield)
                    else:
                        h2RawDict[k].SetBinContent(ptbin_index, ctbin_index, hypYield)
                        h2RawDict[k].SetBinError(ptbin_index, ctbin_index, eYield)

    # write on file
    cent_dir.cd()
    h2BDTeff.Write()
    h2SelEff.Write()

    for h2Raw in h2RawCounts:
        h2Raw.Write()
    if params['BDT_EFF_CUTS']:
        for dictionary in h2RawCountsFixEffDict:
            for th2 in dictionary.values():
                th2.Write()

results_file.Close()

if not params['LOAD_SCORE_EFF']:
    with open(score_bdteff_name, 'w') as out_file:
        yaml.safe_dump(score_bdteff_dict, out_file, default_flow_style=False)
# print execution time to performance evaluation
print('')
print('--- {:.4f} minutes ---'.format((time.time() - start_time) / 60))
