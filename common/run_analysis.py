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

from ROOT import TFile, TF1, TH1D, TH2D, TH3D, TCanvas, TPaveText, gStyle, gROOT

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

if params['FIXED_SIGMA_FIT']:
    h3_invmassptct = TH3D(
        'InvMassPtCt', ';#it{M} (^{3}He + #pi^{-}) (GeV/#it{c}^{2});#it{p}_{T} (GeV/#it{c});c#it{t} (cm)', 40, np.
        array(np.arange(2.96, 3.05225, 0.00225),
              'double'),
        len(params['PT_BINS']) - 1, np.array(params['PT_BINS'],
                                             'double'),
        len(params['CT_BINS']) - 1, np.array(params['CT_BINS'],
                                             'double'))
    h2sigma_mc = TH2D('MCsigmas', ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);#sigma',
                      len(params['PT_BINS'])-1, np.array(
                          params['PT_BINS'], 'double'), len(params['CT_BINS']) - 1,
                      np.array(params['CT_BINS'], 'double'))

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
    backgound_selection = '{}<=HypCandPt<={}'.format(
        params['PT_BINS'][0], params['PT_BINS'][-1])

split = 0

if args.anti:
    split = 'a'
if args.matter:
    split = 'm'

# bkgReservedFraction = params['DEDICATED_BACKGROUND'] if 'DEDICATED_BACKGROUND' in params else 0

# initilize the analysis object
analysis = GeneralizedAnalysis(params['NBODY'], mc_path, data_path,
                               signal_selection, backgound_selection,
                               cent_class=params['CENTRALITY_CLASS'], split=split,
                               dedicated_background=0, training_columns=params['TRAINING_COLUMNS'])

# start timer for performance evaluation
start_time = time.time()

# params for config the analysis
hyperparams = params['HYPERPARAMS_RANGE'] if args.optimize else params['HYPERPARAMS']
optimisation_strategy = 'gs' if params['OPTIMIZATION_STRATEGY'] == 'gs' else 'bayes'


file_name = results_dir + '/' + params['FILE_PREFIX'] + '_results.root'
results_file = TFile(file_name, 'recreate')


for cclass in params['CENTRALITY_CLASS']:
    cent_dir = results_file.mkdir('{}-{}'.format(cclass[0], cclass[1]))

    # create the histos for storing analysis stuff
    h2BDTeff = au.h2_bdteff(params['PT_BINS'], params['CT_BINS'])
    h2seleff = au.h2_seleff(params['PT_BINS'], params['CT_BINS'])

    bkg_models = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']
    fit_directories = []
    h2raw_counts = []
    h2significance = []
    h2raw_counts_fixeff_dict = []
    for model in bkg_models:
        fit_directories.append(cent_dir.mkdir(model))

        if params['BDT_EFF_CUTS']:
            mydict = {}

            for fix_eff in params['BDT_EFFICIENCY']:
                mydict['eff{}'.format(fix_eff)] = au.h2_rawcounts(
                    params['PT_BINS'], params['CT_BINS'], 'RawCounts{}_{}'.format(fix_eff, model))

            h2raw_counts_fixeff_dict.append(mydict)

        h2raw_counts.append(au.h2_rawcounts(params['PT_BINS'], params['CT_BINS'], 'RawCounts_{}'.format(model)))
        h2significance.append(au.h2_rawcounts(params['PT_BINS'], params['CT_BINS'], f'significance_{model}'))

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
                cclass, ct_range=ctbin, pt_range=ptbin, test=args.test, sig_nocent=False)

            # train the models if required or load trained models
            if args.train:
                # train and test the model with some performance plots
                model = analysis.train_test_model(
                    data, params['TRAINING_COLUMNS'], params['XGBOOST_PARAMS'],
                    hyperparams=hyperparams, ct_range=ctbin,
                    cent_class=cclass, pt_range=ptbin, optimize=args.optimize,
                    optimize_mode=optimisation_strategy)

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
            h2seleff.SetBinContent(ptbin_index, ctbin_index, preselection_efficiency[key])

            # compute and store score cut for fixed efficiencies, if required
            if params['BDT_EFF_CUTS'] and not params['LOAD_SCORE_EFF']:
                score_eff = analysis.score_from_efficiency(
                    model, data[2: 4],
                    params['BDT_EFFICIENCY'],
                    params['TRAINING_COLUMNS'],
                    ct_range=ctbin, pt_range=ptbin, cent_class=cclass)

                for se in score_eff:
                    score_bdteff_dict[key]['eff{}'.format(se[1])] = [float(se[0]), float(se[1])]

            if params['FIXED_SIGMA_FIT']:
                data[2]['Score']=data[2]['Score'].astype(float)
                df_mcselected = data[2].query('y > 0.5 and Score > {}'.format(score_cut))

                for _, hyp in df_mcselected.iterrows():
                    h3_invmassptct.Fill(hyp['InvMass'], hyp['HypCandPt'], hyp['ct'])

                del df_mcselected

                mc_minv = h3_invmassptct.ProjectionX('mc_minv', ptbin_index, ptbin_index, ctbin_index, ctbin_index)
                mc_minv.Fit('gaus')

                gaus_fit = mc_minv.GetFunction('gaus')
                if gaus_fit:
                    h2sigma_mc.SetBinContent(ptbin_index, ctbin_index, gaus_fit.GetParameter(2))
                    h2sigma_mc.SetBinError(ptbin_index, ctbin_index, gaus_fit.GetParError(2))

            total_cut = '{}<ct<{} and {}<HypCandPt<{} and {}<centrality<{}'.format(
                ctbin[0], ctbin[1], ptbin[0], ptbin[1], cclass[0], cclass[1])

            df_data = analysis.df_data_all.query(total_cut)

            y_pred = model.predict(df_data[params['TRAINING_COLUMNS']], output_margin=True)
            df_data.eval('Score = @y_pred', inplace=True)

            # extract the signal for each bdtscore-eff configuration
            for k, se in score_bdteff_dict[key].items():
                # obtain the selected invariant mass dist
                mass_bins = 40 if ctbin[1] < 16 else 36

                for model, fitdir, h2raw, h2sig, h2raw_dict in zip(
                        bkg_models, fit_directories, h2raw_counts, h2significance, h2raw_counts_fixeff_dict):
                    massArray = np.array(df_data.query('Score >@se[0]')['InvMass'].values, dtype=np.float64)
                    counts, bins = np.histogram(massArray, bins=mass_bins, range=[2.96, 3.05])

                    # au.fitUnbinned(massArray, ctbin, ptbin, cclass, fitdir)
                    if params['FIXED_SIGMA_FIT']:
                        sigma = h2sigma_mc.GetBinContent(ptbin_index, ctbin_index)
                    else:
                         sigma = -1

                    hyp_yield, err_yield, signif, errsignif, sigma, sigmaErr = au.fit(
                        counts, ctbin, ptbin, cclass, fitdir, name=k, bins=mass_bins, model=model,
                        fixsigma=sigma)

                    if k is 'sig_scan':
                        h2raw.SetBinContent(ptbin_index, ctbin_index, hyp_yield)
                        h2raw.SetBinError(ptbin_index, ctbin_index, err_yield)
                        h2sig.SetBinContent(ptbin_index, ctbin_index, signif)
                        h2sig.SetBinError(ptbin_index, ctbin_index, errsignif)
                    else:
                        h2raw_dict[k].SetBinContent(ptbin_index, ctbin_index, hyp_yield)
                        h2raw_dict[k].SetBinError(ptbin_index, ctbin_index, err_yield)

    # write on file
    cent_dir.cd()
    h2BDTeff.Write()
    h2seleff.Write()
    if params['FIXED_SIGMA_FIT']:
        h3_invmassptct.Write()
        h2sigma_mc.Write()

    for h2raw, h2sig in zip(h2raw_counts, h2significance):
        h2raw.Write()
        h2sig.Write()
    if params['BDT_EFF_CUTS']:
        for dictionary in h2raw_counts_fixeff_dict:
            for th2 in dictionary.values():
                th2.Write()

results_file.Close()

if not params['LOAD_SCORE_EFF']:
    with open(score_bdteff_name, 'w') as out_file:
        yaml.safe_dump(score_bdteff_dict, out_file, default_flow_style=False)
# print execution time to performance evaluation
print('')
print('--- {:.4f} minutes ---'.format((time.time() - start_time) / 60))
