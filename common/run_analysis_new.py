#!/usr/bin/env python3
import argparse
import os
import time
import warnings
from hipe4ml import analysis_utils, plot_utils
from hipe4ml.model_handler import ModelHandler
from analysis_classes import TrainingAnalysis, ModelApplication
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from ROOT import TFile, gROOT
import hyp_analysis_utils as hau

warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()


# ---------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='Do the training',
                    action='store_true')
parser.add_argument(
    '--test', help='Just test the functionalities (training with reduced number of candidates)', action='store_true')
parser.add_argument('-o', '--optimize',
                    help='Run the optimization', action='store_true')
parser.add_argument('-s', '--significance',
                    help='Run the significance optimisation studies', action='store_true')
parser.add_argument('-split', '--split',
                    help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('config', help='Path to the YAML configuration file')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
bkg_path = os.path.expandvars(params['BKG_PATH'])
data_path = os.path.expandvars(params['DATA_PATH'])
results_dir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]
# -----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
if args.split:
    split_list = ['_matter', '_antimatter']
else:
    split_list = ['']
# ----------------------------------------------------------------------------------

start_time = time.time()

if args.train:
    for split_type in split_list:

        for cclass in params['CENTRALITY_CLASS']:

            ml_analysis = TrainingAnalysis(
                params['NBODY'], signal_path, bkg_path, split_type)

            ml_analysis.compute_preselection_efficiency(
                cclass, params['CT_BINS'], params['PT_BINS'], split_type)

            for ptbin in zip(params['PT_BINS'][:-1], params['PT_BINS'][1:]):

                for ctbin in zip(params['CT_BINS'][:-1], params['CT_BINS'][1:]):

                    print('============================================')
                    print('centrality: ', cclass, ' ct: ', ctbin,
                          ' pT: ', ptbin, ' split: ', split_type)
                    part_time = time.time()

                    # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
                    data = ml_analysis.prepare_dataframe(
                        params['TRAINING_COLUMNS'],
                        cclass, ct_range=ctbin, pt_range=ptbin)

                    input_model = xgb.XGBClassifier()
                    model_handler = ModelHandler(input_model)
                    model_handler.set_model_params(params['XGBOOST_PARAMS'])
                    model_handler.set_model_params(params['HYPERPARAMS'])
                    model_handler.set_training_columns(
                        params['TRAINING_COLUMNS'])

                    hyp_ranges = params['HYPERPARAMS_RANGE']
                    if args.optimize:
                        model_handler.optimize_params_bayes(
                            data, hyp_ranges, 'roc_auc')

                    model_handler.train_test_model(data)

                    print(
                        '--- model trained in {:.4f} minutes ---\n'.format((time.time() - part_time) / 60))

                    y_pred = model_handler.predict(data[2])
                    data[2]['Score'] = y_pred

                    _max, _min, _step = params['BDT_EFFICIENCY']
                    efficiency_score_array = ml_analysis.compute_BDT_efficiency(
                        data[3], y_pred, np.arange(_max, _min, _step))

                    if params['FIXED_SIGMA_FIT']:
                        ml_analysis.compute_and_save_MC_sigma_array(
                            data, efficiency_score_array, cclass, ptbin, ctbin, split_type)

                    ml_analysis.save_ML_analysis(
                        model_handler, efficiency_score_array, cent_class=cclass, pt_range=ptbin, ct_range=ctbin, split_string=split_type)

                    # ml_analysis.save_ML_plots(model_handler, data, efficiency_score_array, cent_class=cclass, pt_range=ptbin,
                    #                           ct_range=ctbin, split_string=split_type)
    del ml_analysis
    print('')
    print('--- {:.4f} minutes ---'.format((time.time() - start_time) / 60))


if args.significance:

    bkg_models = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']

    for split_type in split_list:

        file_name = results_dir + '/' + \
            params['FILE_PREFIX'] + '_results' + split_type + '.root'
        results_file = TFile(file_name, 'recreate')
        ml_application = ModelApplication(
            params['NBODY'], data_path, params['CENTRALITY_CLASS'], split_type)

        for cclass in params['CENTRALITY_CLASS']:

            # create output structure--------------------------------------------------
            fit_directories_list = []
            fit_directories_sign_scan_list = []
            sign_scan_raw_counts_list = []
            cent_dir = results_file.mkdir('{}-{}'.format(cclass[0], cclass[1]))
            for model in bkg_models:
                fit_directories_list.append(cent_dir.mkdir(model))
                fit_directories_sign_scan_list.append(fit_directories_list[-1].mkdir('fit_sign_scan'))
                hist_name = 'RawCounts' + '_' + model
                sign_scan_raw_counts_list.append(hau.h2_rawcounts(
                    params['PT_BINS'], params['CT_BINS'], hist_name))

            raw_counts_list = []
            eff_array = np.arange(params['BDT_EFFICIENCY'][0], params['BDT_EFFICIENCY'][1],
                                  params['BDT_EFFICIENCY'][2])
            for eff in np.round(eff_array, 2):
                for model in bkg_models:
                    hist_name = 'RawCounts_' + '%.*f' % (2, eff) + '_' + model
                    raw_counts_list.append(hau.h2_rawcounts(
                        params['PT_BINS'], params['CT_BINS'], hist_name))

            bdt_efficiency_sig_scan_histo = hau.h2_rawcounts(
                params['PT_BINS'], params['CT_BINS'], 'BDTeff')
            # --------------------------------------------------------------------------

            ml_application.load_preselection_efficiency(cclass, split_type)

            for ptbin in zip(params['PT_BINS'][:-1], params['PT_BINS'][1:]):
                ptbin_index = raw_counts_list[0].GetXaxis().FindBin(
                    0.5 * (ptbin[0] + ptbin[1]))

                for ctbin in zip(params['CT_BINS'][:-1], params['CT_BINS'][1:]):
                    ctbin_index = raw_counts_list[0].GetYaxis().FindBin(
                        0.5 * (ctbin[0] + ctbin[1]))

                    print('\n============================================')
                    print('centrality: ', cclass, ' ct: ', ctbin,
                          ' pT: ', ptbin, ' split: ', split_type)

                    pres_eff = ml_application.return_preselection_efficiency(
                        ptbin_index, ctbin_index)
                    efficiency_score_array, model_handler = ml_application.load_ML_analysis(
                        cclass, ptbin, ctbin, split_type)

                    df_skimmed = ml_application.apply_BDT_to_data(
                        model_handler, cclass, ptbin, ctbin, model_handler.get_training_columns())

                    sign_scan_BDT_eff, sign_scan_score = ml_application.significance_scan(df_skimmed,
                                                                                          pres_eff, efficiency_score_array, cclass, ctbin, ptbin)
                    bdt_efficiency_sig_scan_histo.SetBinContent(
                        ptbin_index, ctbin_index, sign_scan_BDT_eff)

                    if params['FIXED_SIGMA_FIT']:
                        fixed_sigma_array = ml_application.load_sigma_array(
                            cclass, ptbin, ctbin, split_type)
                    else:
                        fixed_sigma_array = -1 * \
                            np.ones(len(efficiency_score_array[0]))

                    mass_bins = 40 if ctbin[1] < 16 else 36

                    print("\nFitting at fixed efficiencies ...")
                    for eff_index, (efficiency, cut_score, sigma) in enumerate(zip(efficiency_score_array[0], efficiency_score_array[1], fixed_sigma_array)):

                        massArray = np.array(df_skimmed.query('Score >@cut_score')[
                                             'InvMass'].values, dtype=np.float64)
                        counts, bins = np.histogram(
                            massArray, bins=mass_bins, range=[2.96, 3.05])
                        for bkg_index, (fit_dir, model) in enumerate(zip(fit_directories_list, bkg_models)):
                            fit_name = 'cv_cent{}{}_ct{}{}_pt{}{}_eff{}'.format(cclass[0], cclass[1], ctbin[0], ctbin[1],
                                                                                ptbin[0], ptbin[1], round(efficiency, 2))
                            if efficiency == sign_scan_BDT_eff:
                                fit_dir = fit_directories_sign_scan_list[bkg_index]

                            hyp_yield, err_yield, signif, errsignif, sigma, sigmaErr = hau.fit(counts, ctbin, ptbin, cclass, fit_dir,
                                                                                               name=fit_name, bins=mass_bins, model=model, fixsigma=sigma)
                            raw_counts_list[eff_index*(len(bkg_models))+bkg_index].SetBinContent(
                                ptbin_index, ctbin_index, hyp_yield)

                            if efficiency == sign_scan_BDT_eff:
                                sign_scan_raw_counts_list[bkg_index].SetBinContent(
                                    ptbin_index, ctbin_index, hyp_yield)

                    print('Fits: Done!')

            cent_dir.cd()
            bdt_efficiency_sig_scan_histo.Write()
            ml_application.presel_histo.Write()
            for raw_counts_histo in raw_counts_list:
                raw_counts_histo.Write()

        results_file.Close()
