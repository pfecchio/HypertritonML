import math
import os
from concurrent.futures import ThreadPoolExecutor
from math import floor, log10

import aghast
import numpy as np
import pandas as pd
import ROOT
import uproot
import xgboost as xgb
from hipe4ml.model_handler import ModelHandler
from ROOT import TF1, TH1D, TH2D, TH3D, TCanvas, TPaveStats, TPaveText, gStyle


def get_applied_mc(mc_path, cent_classes, pt_bins, ct_bins, training_columns, application_columns, mode=2, split=''):
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('\nStarting BDT appplication on MC data')

    if mode == 3:
        handlers_path = os.environ['HYPERML_MODELS_3'] + '/handlers'
        efficiencies_path = os.environ['HYPERML_EFFICIENCIES_3']

    if mode == 2:
        handlers_path = os.environ['HYPERML_MODELS_2'] + '/handlers'
        efficiencies_path = os.environ['HYPERML_EFFICIENCIES_2']

    df_signal = uproot.open(mc_path)['SignalTable'].arrays(library='pd')
    df_applied = pd.DataFrame()

    for cclass in cent_classes:
        for ptbin in zip(pt_bins[:-1], pt_bins[1:]):
            for ctbin in zip(ct_bins[:-1], ct_bins[1:]):
                info_string = '_{}{}_{}{}_{}{}'.format(cclass[0], cclass[1], ptbin[0], ptbin[1], ctbin[0], ctbin[1])

                filename_handler = handlers_path + '/model_handler' + info_string + split + '.pkl'
                filename_efficiencies = efficiencies_path + '/Eff_Score' + info_string + split + '.npy'

                model_handler = ModelHandler()
                model_handler.load_model_handler(filename_handler)

                eff_score_array = np.load(filename_efficiencies)
                tsd = eff_score_array[1][-1]

                data_range = f'{ctbin[0]}<ct<{ctbin[1]} and {ptbin[0]}<pt<{ptbin[1]} and {cclass[0]}<=centrality<{cclass[1]}'

                df_tmp = df_signal.query(data_range)
                df_tmp.insert(0, 'score', model_handler.predict(df_tmp[training_columns]))

                df_tmp = df_tmp.query('score>@tsd')
                df_tmp = df_tmp.loc[:, application_columns]

                df_applied = df_applied.append(df_tmp, ignore_index=True, sort=False)

    print(df_applied.info(memory_usage='deep'))
    return df_applied


def get_skimmed_data(data_path, cent_classes, pt_bins, ct_bins, training_columns, application_columns, mode=2, split='', chunks=False):
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('\nStarting BDT appplication on large data')

    if mode == 3:
        handlers_path = os.environ['HYPERML_MODELS_3'] + '/handlers'
        efficiencies_path = os.environ['HYPERML_EFFICIENCIES_3']

    if mode == 2:
        handlers_path = os.environ['HYPERML_MODELS_2'] + '/handlers'
        efficiencies_path = os.environ['HYPERML_EFFICIENCIES_2']
    
    if not data_path[-5:]==".root":
        data_iterator = [pd.read_parquet(data_path, engine='fastparquet')]
    else:
        if chunks:
            executor = ThreadPoolExecutor()
            data_iterator = uproot.iterate(f'{data_path}:DataTable', library='pd', executor=executor)
        else:
            data_iterator = [uproot.open(data_path)['DataTable'].arrays(library='pd')]

    results = []
    for data in data_iterator:
        
        for cclass in cent_classes:
            for ptbin in zip(pt_bins[:-1], pt_bins[1:]):
                for ctbin in zip(ct_bins[:-1], ct_bins[1:]):
                    info_string = '_{}{}_{}{}_{}{}'.format(cclass[0], cclass[1], ptbin[0], ptbin[1], ctbin[0], ctbin[1])

                    filename_handler = handlers_path + '/model_handler' + info_string + split + '.pkl'
                    filename_efficiencies = efficiencies_path + '/Eff_Score' + info_string + split + '.npy'

                    model_handler = ModelHandler()
                    model_handler.load_model_handler(filename_handler)

                    eff_score_array = np.load(filename_efficiencies)
                    tsd = eff_score_array[1][-1]

                    data_range = f'{ctbin[0]}<ct<{ctbin[1]} and {ptbin[0]}<pt<{ptbin[1]} and {cclass[0]}<=centrality<{cclass[1]}'

                    df_tmp = data.query(data_range)
                    df_tmp.insert(0, 'score', model_handler.predict(df_tmp[training_columns]))

                    df_tmp = df_tmp.query('score>@tsd')
                    df_tmp = df_tmp[application_columns]

                    results.append(df_tmp)


    results = pd.concat(results)

    print(results.info(memory_usage='deep'))
    return results
    

def expected_signal_counts(bw, cent_range, pt_range, eff, nevents, n_body=2):
    correction = 0.4  # Very optimistic, considering it constant with centrality

    if n_body == 2:
        correction *= 0.25
        
    if n_body == 3:
        correction *= 0.4

    cent_bins = [10, 40, 90]

    signal = 0
    for cent in range(cent_range[0]+1, cent_range[1]):
        for index in range(0, 3):
            if cent < cent_bins[index]:
                signal = signal + \
                    nevents[cent] * \
                    bw[index].Integral(pt_range[0], pt_range[1], 1e-8)
                break

    return int(round(2*signal * eff * correction))


def expo(x, tau):
    return np.exp(-x / (tau * 0.029979245800))


def h_weighted_average(histo):
    aver = 0    
    weights = 0
    for iBin in range(1,histo.GetNbinsX()+1):
        counts = histo.GetBinContent(iBin)
        err = histo.GetBinError(iBin)
        aver += counts*(1/err**2)
        weights += (1/err**2)
    aver = aver/weights
    error = np.sqrt(1/weights)
    return aver, error

def h2_preselection_efficiency(ptbins, ctbins, name='PreselEff'):
    th2 = TH2D(name, ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Preselection efficiency',
               len(ptbins) - 1, np.array(ptbins, 'double'), len(ctbins) - 1, np.array(ctbins, 'double'))
    th2.SetDirectory(0)

    return th2


def h2_generated(ptbins, ctbins, name='Generated'):
    th2 = TH2D(name, ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm); Generated', len(ptbins)-1,
               np.array(ptbins, 'double'), len(ctbins) - 1, np.array(ctbins, 'double'))
    th2.SetDirectory(0)

    return th2


def h2_rawcounts(ptbins, ctbins, name='RawCounts', suffix=''):
    th2 = TH2D(f'{name}{suffix}', ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Raw counts', len(ptbins)-1,
               np.array(ptbins, 'double'), len(ctbins) - 1, np.array(ctbins, 'double'))
    th2.SetDirectory(0)

    return th2


def h2_significance(ptbins, ctbins, name='Significance', suffix=''):
    th2 = TH2D(f'{name}{suffix}', ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Significance', len(ptbins)-1,
               np.array(ptbins, 'double'), len(ctbins) - 1, np.array(ctbins, 'double'))
    th2.SetDirectory(0)

    return th2


def round_to_error(x, error):
    return round(x, -int(floor(log10(abs(error)))))


def get_ptbin_index(th2, ptbin):
    return th2.GetXaxis().FindBin(0.5 * (ptbin[0] + ptbin[1]))


def get_ctbin_index(th2, ctbin):
    return th2.GetYaxis().FindBin(0.5 * (ctbin[0] + ctbin[1]))



def ndarray2roo(ndarray, var):
    if isinstance(ndarray, ROOT.RooDataSet):
        print('Already a RooDataSet')
        return ndarray

    assert isinstance(ndarray, np.ndarray), 'Did not receive NumPy array'
    assert len(ndarray.shape) == 1, 'Can only handle 1d array'

    name = var.GetName()
    x = np.zeros(1, dtype=np.float64)

    tree = ROOT.TTree('tree', 'tree')
    tree.Branch(f'{name}', x ,f'{name}/D')

    for i in ndarray:
        x[0] = i
        tree.Fill()

    array_roo = ROOT.RooDataSet('data', 'dataset from tree', tree, ROOT.RooArgSet(var))
    return array_roo


def significance_error(signal, background):
    signal_error = np.sqrt(signal + 1e-10)
    background_error = np.sqrt(background + 1e-10)

    sb = signal + background + 1e-10
    sb_sqrt = np.sqrt(sb)

    s_propag = (sb_sqrt + signal / (2 * sb_sqrt))/sb * signal_error
    b_propag = signal / (2 * sb_sqrt)/sb * background_error

    if signal+background == 0:
        return 0

    return np.sqrt(s_propag * s_propag + b_propag * b_propag)




def compute_significance(roodataset, mass, signal, background, n, mu = None, sigma = None):
    if mu==None and sigma==None:
        # fit with gaussian for the 3 sigma region determination only
        mu = ROOT.RooRealVar('mu', 'hypertriton mass', 2.989, 2.993, 'GeV/c^{2}')
        sigma = ROOT.RooRealVar('sigma', 'hypertriton width', 0.0001, 0.004, 'GeV/c^{2}')
        signal_gauss = ROOT.RooGaussian('signal_gauss', 'signal gauss', mass, mu, sigma)
        fit_function_gauss = ROOT.RooAddPdf('temp_gaus', 'signal + background', ROOT.RooArgList(signal_gauss, background), ROOT.RooArgList(n))
        fit_function_gauss.fitTo(roodataset, ROOT.RooFit.Range(2.960, 3.040), ROOT.RooFit.NumCPU(32), ROOT.RooFit.Save())
        mu.setConstant(ROOT.kTRUE)
        mu.removeError()
        sigma.setConstant(ROOT.kTRUE)
        sigma.removeError()

    # compute signal and significance
    mass.setRange('3sigma', mu.getVal() - 3*sigma.getVal(), mu.getVal() + 3*sigma.getVal())
    mass_set = ROOT.RooArgSet(mass)
    mass_norm_set = ROOT.RooFit.NormSet(mass_set)
    frac_signal_range = signal.createIntegral(mass_set, mass_norm_set, ROOT.RooFit.Range('3sigma'))
    frac_background_range = background.createIntegral(mass_set, mass_norm_set, ROOT.RooFit.Range('3sigma'))

    sig = n.getVal() * frac_signal_range.getVal() * roodataset.sumEntries()
    sig_error = n.getError() * frac_signal_range.getVal() * roodataset.sumEntries()

    bkg = (1 - n.getVal()) * frac_background_range.getVal() * roodataset.sumEntries()
    bkg_error = n.getError() * frac_background_range.getVal() * roodataset.sumEntries()

    significance = sig / np.sqrt(sig + bkg + 1e-10)
    significance_err = significance_error(sig, bkg)
    return significance, significance_err


def b_form_histo(histo):
    pol0 = ROOT.TF1('blfunction', '1115.683 + 1875.61294257 - [0]', 0, 10)
    histo.Fit(pol0)

    blambda = pol0.GetParameter(0)
    mass = 1115.683 + 1875.61294257 - blambda
    mass_error = pol0.GetParError(0)

    if pol0.GetNDF() == 0:
        chi2red = 100
    else:
        chi2red = pol0.GetChisquare()/pol0.GetNDF()

    return mass, mass_error, chi2red


