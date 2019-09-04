import io
import os
from contextlib import redirect_stdout

import numpy as np

import xgboost as xgb
import math
from ROOT import TF1, TFile, gDirectory
from ROOT import TH2D, TH1D, TCanvas, TFile, TF1, gStyle, TPaveText

# target function for the bayesian hyperparameter optimization


def evaluate_hyperparams(
        dtrain, reg_params, eta, min_child_weight, max_depth, gamma, subsample, colsample_bytree,
        scale_pos_weight, num_rounds=100, es_rounds=2, nfold=3, round_score_list=[]):
    params = {'eval_metric': 'auc',
              'eta': eta,
              'min_child_weight': int(min_child_weight),
              'max_depth': int(max_depth),
              'gamma': gamma,
              'subsample': subsample,
              'colsample_bytree': colsample_bytree,
              'scale_pos_weight': scale_pos_weight}
    params = {**reg_params, **params}

    # Use around 1000 boosting rounds in the full model
    cv_result = xgb.cv(params, dtrain, num_boost_round=num_rounds, early_stopping_rounds=es_rounds, nfold=nfold)

    best_boost_rounds = cv_result['test-auc-mean'].idxmax()
    best_score = cv_result['test-auc-mean'].max()

    round_score_list.append(tuple([best_score, best_boost_rounds]))

    return best_score


def gs_1par(gs_dict, par_dict, train_data, num_rounds, seed, folds, metrics, n_early_stop):

    fp_dict = gs_dict['first_par']
    gs_params = fp_dict['par_values']

    max_auc = 0.
    best_params = None
    for val in gs_params:
        # Update our parameters
        par_dict[fp_dict['name']] = val

        # Run CV
        trap = io.StringIO()
        with redirect_stdout(trap):
            cv_results = xgb.cv(par_dict, train_data, num_boost_round=num_rounds, seed=seed,
                                folds=folds, metrics=metrics, early_stopping_rounds=n_early_stop)

        # Update best AUC
        mean_auc = cv_results['test-auc-mean'].max()
        boost_rounds = cv_results['test-auc-mean'].idxmax()
        mean_std = cv_results['test-auc-std'][boost_rounds]
        if mean_auc > max_auc:
            max_auc = mean_auc
            max_std = mean_std
            best_params = (val, boost_rounds)
    return (best_params)


def gs_2par(gs_dict, par_dict, train_data, num_rounds, seed, folds, metrics, n_early_stop):

    fp_dict = gs_dict['first_par']
    sp_dict = gs_dict['second_par']
    gs_params = [(first_val, second_val)
                 for first_val in fp_dict['par_values']
                 for second_val in sp_dict['par_values']
                 ]

    max_auc = 0.
    best_params = None
    for first_val, second_val in gs_params:
        # Update our parameters
        par_dict[fp_dict['name']] = first_val
        par_dict[sp_dict['name']] = second_val

        # Run CV
        trap = io.StringIO()
        with redirect_stdout(trap):
            cv_results = xgb.cv(par_dict, train_data, num_boost_round=num_rounds, seed=seed,
                                folds=folds, metrics=metrics, early_stopping_rounds=n_early_stop)

        # Update best AUC
        mean_auc = cv_results['test-auc-mean'].max()
        boost_rounds = cv_results['test-auc-mean'].idxmax()
        mean_std = cv_results['test-auc-std'][boost_rounds]
        if mean_auc > max_auc:
            max_auc = mean_auc
            max_std = mean_std
            best_params = (first_val, second_val, boost_rounds)
    return (best_params)


def expected_signal_raw(pt_range, cent_bin):
    bw_ = os.environ['HYPERML_UTILS']
    bw_file = TFile(os.environ['HYPERML_UTILS'] + '/BlastWaveFits.root')

    scale_factor = [3.37e-5, 1.28e-5, 0.77e-5, 0.183e-5]
    cent_ref = [[0, 10], [10, 30], [30, 50], [50, 90]]
    cent_class = cent_ref.index(cent_bin)
    if cent_class == 2:
        bw1 = bw_file.Get("BlastWave/BlastWave1")
        bw2 = bw_file.Get("BlastWave/BlastWave2")

        bw1_integral_tot = bw1.Integral(0, 10, 1e-8)
        bw2_integral_tot = bw2.Integral(0, 10, 1e-8)

        bw_integral_tot = bw1_integral_tot + bw2_integral_tot

        bw1_integral_range = bw1.Integral(pt_range[0], pt_range[1], 1e-8)
        bw2_integral_range = bw2.Integral(pt_range[0], pt_range[1], 1e-8)

        bw_integral_range = bw1_integral_range + bw2_integral_range

    else:
        if cent_class == 3:
            cent_class = 2

        bw = bw_file.Get('BlastWave/BlastWave{}'.format(cent_class))

        bw_integral_tot = bw.Integral(0, 10, 1e-8)
        bw_integral_range = bw.Integral(pt_range[0], pt_range[1], 1e-8)

    pt_width = pt_range[1] - pt_range[0]

    exp_yield = 2 * scale_factor[cent_class] * bw_integral_range / pt_width / bw_integral_tot

    return exp_yield


def expected_signal_counts(pt_range, eff, cent_range, nevents):  # nevents assumed to be the number of events in 1% bins
    # TODO: this function is called in a closed loop, we cannot open the file everytime...
    hyp2he3 = 0.4 * 0.25  # Very optimistic, considering it constant with centrality
    bw_file = TFile(os.environ['HYPERML_UTILS'] + '/BlastWaveFits.root')
    bw = [bw_file.Get("BlastWave/BlastWave{}".format(i)) for i in [0, 1, 2]]
    cent_bins = [10, 40, 90]

    signal = 0
    for cent in range(cent_range[0]+1, cent_range[1]):
        for index in range(0, 3):
            if cent < cent_bins[index]:
                signal = signal + nevents[cent] * bw[index].Integral(pt_range[0], pt_range[1], 1e-8)
                break

    return int(round(signal * eff * hyp2he3))


def expected_signal(n_ev, eff_presel, eff_bdt, pt_range, cent_class):
    signal_raw = expected_signal_raw(pt_range, cent_class)

    return int(round(n_ev * signal_raw * (pt_range[1] - pt_range[0]) * eff_presel * eff_bdt))


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


def expo(x, tau):
    return np.exp(-x / tau / 0.029979245800)


def fit(counts, min, max, nsigma=3, recreate=False, signif=0, errsignif=0, minCent=0, maxCent=90, filename='results.root'):
    if recreate is True:
        results = TFile(os.environ['HYPERML_DATA_2']+'/'+filename, "RECREATE")
    else:
        results = TFile(os.environ['HYPERML_DATA_2']+'/'+filename, "UPDATE")

    histo = TH1D("histo{}_{}".format(min, max), ";ct[cm];dN/dct [cm^{-1}]", 45, 2.96, 3.05)
    for index in range(0, len(counts)):
        histo.SetBinContent(index+1, counts[index])
        histo.SetBinError(index+1, math.sqrt(counts[index]))

    results.cd()
    cv = TCanvas("cv{}_{}".format(min, max))
    fitTpl = TF1("fitTpl", "pol2(0)+gausn(3)", 0, 5)
    fitTpl.SetParNames("B_{0}", "B_{1}", "B_{2}", "N_{sig}", "#mu", "#sigma")
    bkgTpl = TF1("fitTpl", "pol2(0)", 0, 5)
    sigTpl = TF1("fitTpl", "gausn(0)", 0, 5)
    fitTpl.SetNpx(300)
    fitTpl.SetLineWidth(2)
    fitTpl.SetLineColor(2)
    bkgTpl.SetNpx(300)
    bkgTpl.SetLineWidth(2)
    bkgTpl.SetLineStyle(2)
    bkgTpl.SetLineColor(2)

    fitTpl.SetParameter(3, 40)
    fitTpl.SetParameter(4, 2.991)
    fitTpl.SetParLimits(4, 2.99, 3)
    fitTpl.SetParameter(5, 0.002)
    fitTpl.SetParLimits(5, 0.0001, 0.004)

    gStyle.SetOptStat(0)
    gStyle.SetOptFit(0)
    ####################

    histo.UseCurrentStyle()
    histo.SetLineColor(1)
    histo.SetMarkerStyle(20)
    histo.SetMarkerColor(1)
    histo.SetTitle(";m (^{3}He + #pi) (GeV/#it{c})^{2};Counts / 2 MeV")
    histo.SetMaximum(1.5 * histo.GetMaximum())
    histo.Fit(fitTpl, "QRM", "", 2.97, 3.03)
    histo.Fit(fitTpl, "QRM", "", 2.97, 3.03)
    histo.SetDrawOption("e")
    histo.GetXaxis().SetRangeUser(2.97, 3.02)
    bkgTpl.SetParameters(fitTpl.GetParameters())
    bkgTpl.SetLineColor(600)
    bkgTpl.SetLineStyle(2)
    bkgTpl.Draw("same")
    sigTpl.SetParameter(0, fitTpl.GetParameter(3))
    sigTpl.SetParameter(1, fitTpl.GetParameter(4))
    sigTpl.SetParameter(2, fitTpl.GetParameter(5))
    sigTpl.SetLineColor(600)
    # sigTpl.Draw("same")
    mu = fitTpl.GetParameter(4)
    sigma = fitTpl.GetParameter(5)
    signal = fitTpl.GetParameter(3) / histo.GetBinWidth(1)
    errsignal = fitTpl.GetParError(3) / histo.GetBinWidth(1)
    bkg = bkgTpl.Integral(mu - nsigma * sigma, mu + nsigma * sigma) / histo.GetBinWidth(1)

    if bkg > 0:
        errbkg = math.sqrt(bkg)
    else:
        errbkg = 0

    peak = histo.Integral(int(len(counts) * (mu - nsigma * sigma - 2.96) / (3.05 - 2.96)),
                          int(len(counts) * (mu + nsigma * sigma - 2.96) / (3.05 - 2.96)))

    NHyTr = (peak-bkg)
    print(peak, ' ', bkg)
    if peak+bkg > 0 and signal+bkg > 0:
        ErrNHyTr = math.sqrt(peak+bkg)
        signif = signal/math.sqrt(signal+bkg)
        deriv_sig = 1/math.sqrt(signal+bkg)-signif/(2*(signal+bkg))
        deriv_bkg = -signal/(2*(math.pow(signal+bkg, 1.5)))
        errsignif = math.sqrt((errsignal*deriv_sig)**2+(errbkg*deriv_bkg)**2)
    else:
        print('sig+bkg<0')
        ErrNHyTr = 0
        signif = 0
        errsignif = 0

    pinfo2 = TPaveText(0.5, 0.5, 0.91, 0.9, "NDC")
    pinfo2.SetBorderSize(0)
    pinfo2.SetFillStyle(0)
    pinfo2.SetTextAlign(30+3)
    pinfo2.SetTextFont(42)
    string = 'ALICE Internal, Pb-Pb 2018 {}-{}%'.format(minCent, maxCent)
    pinfo2.AddText(string)
    string = '{}^{3}_{#Lambda}H#rightarrow ^{3}He#pi + c.c., %i #leq #it{ct} < %i cm ' % (min, max)
    pinfo2.AddText(string)
    string = 'Significance ({:.0f}#sigma) {:.1f} #pm {:.1f} '.format(nsigma, signif, errsignif)
    pinfo2.AddText(string)

    string = 'S ({:.0f}#sigma) {:.0f} #pm {:.0f} '.format(nsigma, signal, errsignal)
    pinfo2.AddText(string)
    string = 'B ({:.0f}#sigma) {:.0f} #pm {:.0f}'.format(nsigma, bkg, errbkg)
    pinfo2.AddText(string)
    if bkg > 0:
        ratio = signal/bkg
        string = 'S/B ({:.0f}#sigma) {:.4f} '.format(nsigma, ratio)
    pinfo2.AddText(string)
    pinfo2.Draw()

    histo.Write()
    cv.Write()
    results.Close()
    return (NHyTr, ErrNHyTr)


def Argus(x, *p):
    return p[0]*x*math.pow(1-(x/p[1])**2, p[3])*math.exp(p[2]*(1-(x/p[1])**2))


def write_array(name_file, array, mode):
    file = open(name_file, mode)
    for item in array:
        file.write(str(item)+' ')
        if item == array[len(array)-1]:
            file.write(str(item)+'\n')
    file.close()


def read_array(name_file):
    file = open(name_file, 'r')
    array = []
    string = file.readline()
    for char in string:
        if char is not ' ':
            array.append(char)
    return float(array)
