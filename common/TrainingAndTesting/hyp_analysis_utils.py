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

import hyp_analysis_utils as hau


def get_skimmed_large_data(data_path, cent_classes, pt_bins, ct_bins, training_columns, application_columns, mode, split=''):
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('\nStarting BDT appplication on large data')

    if mode == 3:
        handlers_path = os.environ['HYPERML_MODELS_3'] + '/handlers'
        efficiencies_path = os.environ['HYPERML_EFFICIENCIES_3']

    if mode == 2:
        handlers_path = os.environ['HYPERML_MODELS_2'] + '/handlers'
        efficiencies_path = os.environ['HYPERML_EFFICIENCIES_2']

    executor = ThreadPoolExecutor()
    iterator = uproot.pandas.iterate(data_path, 'DataTable', executor=executor, reportfile=True)

    df_applied = pd.DataFrame()

    for current_file, data in iterator:
        rename_df_columns(data)
    
        print('current file: {}'.format(current_file))
        print ('start entry chunk: {}, stop entry chunk: {}'.format(data.index[0], data.index[-1]))
        
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
                    df_tmp = df_tmp.loc[:, application_columns]

                    df_applied = df_applied.append(df_tmp, ignore_index=True, sort=False)

    print(df_applied.info(memory_usage='deep'))
    return df_applied
    

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
    return np.exp(-x / (tau * 0.029979245800))


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


def h1_invmass(counts, cent_class, pt_range, ct_range, name=''):
    ghist = aghast.from_numpy(counts)
    th1 = aghast.to_root(ghist, f'ct{ct_range[0]}{ct_range[1]}_pT{pt_range[0]}{pt_range[1]}_cen{cent_class[0]}{cent_class[1]}_{name}')
    th1.SetDirectory(0)
    return th1


def round_to_error(x, error):
    return round(x, -int(floor(log10(abs(error)))))


def get_ptbin_index(th2, ptbin):
    return th2.GetXaxis().FindBin(0.5 * (ptbin[0] + ptbin[1]))


def get_ctbin_index(th2, ctbin):
    return th2.GetYaxis().FindBin(0.5 * (ctbin[0] + ctbin[1]))


def fit_hist(
        histo, cent_class, pt_range, ct_range, nsigma=3, model="pol2", fixsigma=-1, sigma_limits=None, mode=3, split =''):
    # canvas for plotting the invariant mass distribution
    cv = TCanvas(f'cv_{histo.GetName()}')

    # define the number of parameters depending on the bkg model
    if 'pol' in str(model):
        n_bkgpars = int(model[3]) + 1
    elif 'expo' in str(model):
        n_bkgpars = 2
    else:
        print(f'Unsupported model {model}')

    # define the fit function bkg_model + gauss
    fit_tpl = TF1('fitTpl', f'{model}(0)+gausn({n_bkgpars})', 0, 5)

    # redefine parameter names for the bkg_model
    for i in range(n_bkgpars):
        fit_tpl.SetParName(i, f'B_{i}')

    # define parameter names for the signal fit
    fit_tpl.SetParName(n_bkgpars, 'N_{sig}')
    fit_tpl.SetParName(n_bkgpars + 1, '#mu')
    fit_tpl.SetParName(n_bkgpars + 2, '#sigma')
    # define parameter values and limits
    fit_tpl.SetParameter(n_bkgpars, 40)
    fit_tpl.SetParLimits(n_bkgpars, 0.001, 10000)
    fit_tpl.SetParameter(n_bkgpars + 1, 2.991)
    fit_tpl.SetParLimits(n_bkgpars + 1, 2.986, 3)

    # define signal and bkg_model TF1 separately
    sigTpl = TF1('fitTpl', 'gausn(0)', 0, 5)
    bkg_tpl = TF1('fitTpl', f'{model}(0)', 0, 5)

    # plotting stuff for fit_tpl
    fit_tpl.SetNpx(300)
    fit_tpl.SetLineWidth(2)
    fit_tpl.SetLineColor(2)
    # plotting stuff for bkg model
    bkg_tpl.SetNpx(300)
    bkg_tpl.SetLineWidth(2)
    bkg_tpl.SetLineStyle(2)
    bkg_tpl.SetLineColor(2)

    # define limits for the sigma if provided
    if sigma_limits != None:
        fit_tpl.SetParameter(n_bkgpars + 2, 0.5 *
                             (sigma_limits[0] + sigma_limits[1]))
        fit_tpl.SetParLimits(n_bkgpars + 2, sigma_limits[0], sigma_limits[1])
    # if the mc sigma is provided set the sigma to that value
    elif fixsigma > 0:
        fit_tpl.FixParameter(n_bkgpars + 2, fixsigma)
    # otherwise set sigma limits reasonably
    else:
        fit_tpl.SetParameter(n_bkgpars + 2, 0.002)
        fit_tpl.SetParLimits(n_bkgpars + 2, 0.001, 0.003)

    ########################################
    # plotting the fits
    if mode == 2:
        ax_titles = ';m (^{3}He + #pi) (GeV/#it{c})^{2};Counts' + f' / {round(1000 * histo.GetBinWidth(1), 2)} MeV'
    if mode == 3:
        ax_titles = ';m (d + p + #pi) (GeV/#it{c})^{2};Counts' + f' / {round(1000 * histo.GetBinWidth(1), 2)} MeV'

    # invariant mass distribution histo and fit
    histo.UseCurrentStyle()
    histo.SetLineColor(1)
    histo.SetMarkerStyle(20)
    histo.SetMarkerColor(1)
    histo.SetTitle(ax_titles)
    histo.SetMaximum(1.5 * histo.GetMaximum())
    histo.Fit(fit_tpl, "QRL", "", 2.96, 3.04)
    histo.Fit(fit_tpl, "QRL", "", 2.96, 3.04)
    histo.SetDrawOption("e")
    histo.GetXaxis().SetRangeUser(2.96, 3.04)
    # represent the bkg_model separately
    bkg_tpl.SetParameters(fit_tpl.GetParameters())
    bkg_tpl.SetLineColor(600)
    bkg_tpl.SetLineStyle(2)
    bkg_tpl.Draw("same")
    # represent the signal model separately
    sigTpl.SetParameter(0, fit_tpl.GetParameter(n_bkgpars))
    sigTpl.SetParameter(1, fit_tpl.GetParameter(n_bkgpars+1))
    sigTpl.SetParameter(2, fit_tpl.GetParameter(n_bkgpars+2))
    sigTpl.SetLineColor(600)
    # sigTpl.Draw("same")

    # get the fit parameters
    mu = fit_tpl.GetParameter(n_bkgpars+1)
    muErr = fit_tpl.GetParError(n_bkgpars+1)
    sigma = fit_tpl.GetParameter(n_bkgpars+2)
    sigmaErr = fit_tpl.GetParError(n_bkgpars+2)
    signal = fit_tpl.GetParameter(n_bkgpars) / histo.GetBinWidth(1)
    errsignal = fit_tpl.GetParError(n_bkgpars) / histo.GetBinWidth(1)
    bkg = bkg_tpl.Integral(mu - nsigma * sigma, mu +
                           nsigma * sigma) / histo.GetBinWidth(1)

    if bkg > 0:
        errbkg = math.sqrt(bkg)
    else:
        errbkg = 0
    # compute the significance
    if signal+bkg > 0:
        signif = signal/math.sqrt(signal+bkg)
        deriv_sig = 1/math.sqrt(signal+bkg)-signif/(2*(signal+bkg))
        deriv_bkg = -signal/(2*(math.pow(signal+bkg, 1.5)))
        errsignif = math.sqrt((errsignal*deriv_sig)**2+(errbkg*deriv_bkg)**2)
    else:
        signif = 0
        errsignif = 0

    # print fit info on the canvas
    pinfo2 = TPaveText(0.5, 0.5, 0.91, 0.9, "NDC")
    pinfo2.SetBorderSize(0)
    pinfo2.SetFillStyle(0)
    pinfo2.SetTextAlign(30+3)
    pinfo2.SetTextFont(42)

    string = f'ALICE Internal, Pb-Pb 2018 {cent_class[0]}-{cent_class[1]}%'
    pinfo2.AddText(string)
    
    decay_label = {
        "": ['{}^{3}_{#Lambda}H#rightarrow ^{3}He#pi^{-} + c.c.','{}^{3}_{#Lambda}H#rightarrow dp#pi^{-} + c.c.'],
        "_matter": ['{}^{3}_{#Lambda}H#rightarrow ^{3}He#pi^{-}','{}^{3}_{#Lambda}H#rightarrow dp#pi^{-}'],
        "_antimatter": ['{}^{3}_{#bar{#Lambda}}#bar{H}#rightarrow ^{3}#bar{He}#pi^{+}','{}^{3}_{#Lambda}H#rightarrow #bar{d}#bar{p}#pi^{+}'],
    }

    string = decay_label[split][mode-2]+', %i #leq #it{ct} < %i cm %i #leq #it{p}_{T} < %i GeV/#it{c} ' % (
        ct_range[0], ct_range[1], pt_range[0], pt_range[1])
    pinfo2.AddText(string)

    string = f'Significance ({nsigma:.0f}#sigma) {signif:.1f} #pm {errsignif:.1f} '
    pinfo2.AddText(string)

    string = f'S ({nsigma:.0f}#sigma) {signal:.0f} #pm {errsignal:.0f}'
    pinfo2.AddText(string)

    string = f'B ({nsigma:.0f}#sigma) {bkg:.0f} #pm {errbkg:.0f}'
    pinfo2.AddText(string)

    if bkg > 0:
        ratio = signal/bkg
        string = f'S/B ({nsigma:.0f}#sigma) {ratio:.4f}'

    pinfo2.AddText(string)
    pinfo2.Draw()
    gStyle.SetOptStat(0)

    st = histo.FindObject('stats')
    if isinstance(st, TPaveStats):
        st.SetX1NDC(0.12)
        st.SetY1NDC(0.62)
        st.SetX2NDC(0.40)
        st.SetY2NDC(0.90)
        st.SetOptStat(0)

    histo.Write()
    cv.Write()

    return (signal, errsignal, signif, errsignif, mu, muErr, sigma, sigmaErr)
    return (signal, errsignal, signif, errsignif, sigma, sigmaErr)


def load_mcsigma(cent_class, pt_range, ct_range, mode, split=''):
    info_string = f'_{cent_class[0]}{cent_class[1]}_{pt_range[0]}{pt_range[1]}_{ct_range[0]}{ct_range[1]}{split}'
    sigma_path = os.environ['HYPERML_UTILS_{}'.format(mode)] + '/FixedSigma'

    file_name = f'{sigma_path}/sigma_array{info_string}.npy'

    return np.load(file_name, allow_pickle=True)


def rename_df_columns(df):
    rename_dict = {}

    for col in df.columns:

        if col.endswith('_f'):
            rename_dict[col] = col[:-2]
    
    df.rename(columns = rename_dict, inplace=True)

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



def unbinned_mass_fit(data, eff, bkg_model, output_dir, cent_class, pt_range, ct_range, split):
    output_dir.cd()

    # define working variable 
    mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2.975, 3.010, 'GeV/c^{2}')

    # define signal parameters
    hyp_mass = ROOT.RooRealVar('hyp_mass', 'hypertriton mass', 2.989, 2.993, 'GeV/c^{2}')
    width = ROOT.RooRealVar('width', 'hypertriton width', 0.001, 0.0025, 'GeV/c^{2}')

    # define signal component
    signal = ROOT.RooGaussian('signal', 'signal component pdf', mass, hyp_mass, width)

    # define background parameters
    slope = ROOT.RooRealVar('slope', 'exponential slope', -100., 100.)

    c0 = ROOT.RooRealVar('c0', 'constant c0', 1.)
    c1 = ROOT.RooRealVar('c1', 'constant c1', 1.)
    c2 = ROOT.RooRealVar('c2', 'constant c2', 1.)

    # define background component depending on background model required
    if bkg_model == 'pol1':
        background = ROOT.RooPolynomial('bkg', 'pol1 bkg', mass, ROOT.RooArgList(c0, c1))
                                        
    if bkg_model == 'pol2':
        background = ROOT.RooPolynomial('bkg', 'pol2 for bkg', mass, ROOT.RooArgList(c0, c1, c2))
        
    if bkg_model == 'expo':
        background = ROOT.RooExponential('bkg', 'expo for bkg', mass, slope)

    # define fraction
    n1 = ROOT.RooRealVar('n1', 'n1 const', 0., 1, 'GeV')

    # define the fit funciton -> signal component + background component
    fit_function = ROOT.RooAddPdf(f'{bkg_model}_gaus', 'signal + background', ROOT.RooArgList(signal, background), ROOT.RooArgList(n1))

    # convert data to RooData               
    roo_data = ndarray2roo(data, mass)

    # fit data
    fit_function.fitTo(roo_data, ROOT.RooFit.Range(2.975, 3.01))

    # plot the fit
    frame = mass.frame(35)

    roo_data.plotOn(frame)
    fit_function.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue))
    fit_function.plotOn(frame, ROOT.RooFit.Components('signal'), ROOT.RooFit.LineStyle(ROOT.kDotted), ROOT.RooFit.LineColor(ROOT.kRed))
    fit_function.plotOn(frame, ROOT.RooFit.Components('bkg'), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed))

    # add info to plot
    nsigma = 3
    mu = hyp_mass.getVal()
    mu_error = hyp_mass.getError()
    sigma = width.getVal()
    sigma_error = width.getError()

    # # compute significance
    # mass.setRange('signal region',  mu - (nsigma * sigma), mu + (nsigma * sigma))
    # signal_counts = int(round(signal.createIntegral(ROOT.RooArgSet(mass), ROOT.RooArgSet(mass), 'signal region').getVal() * n1.getVal()*normalization))
    # background_counts = int(round(background.createIntegral(ROOT.RooArgSet(mass), ROOT.RooArgSet(mass), 'signal region').getVal() * (1 - n1.getVal())*normalization))

    # signif = signal_counts / math.sqrt(signal_counts + background_counts + 1e-10)
    # signif_error = significance_error(signal_counts, background_counts)

    pinfo = ROOT.TPaveText(0.537, 0.474, 0.937, 0.875, 'NDC')
    pinfo.SetBorderSize(0)
    pinfo.SetFillStyle(0)
    pinfo.SetTextAlign(30+3)
    pinfo.SetTextFont(42)
    # pinfo.SetTextSize(12)

    decay_label = {
        '': '{}^{3}_{#Lambda}H#rightarrow ^{3}He#pi^{-} + c.c.',
        '_matter': '{}^{3}_{#Lambda}H#rightarrow ^{3}He#pi^{-}',
        '_antimatter': '{}^{3}_{#bar{#Lambda}}#bar{H}#rightarrow ^{3}#bar{He}#pi^{+}',
    }

    string_list = []

    string_list.append(f'ALICE Internal, Pb-Pb 2018 {cent_class[0]}-{cent_class[1]}%')
    string_list.append(decay_label[split] + ', %i #leq #it{p}_{T} < %i GeV/#it{c} ' % (pt_range[0], pt_range[1]))
    string_list.append(f'#mu {mu*1000:.2f} #pm {mu_error*1000:.2f} MeV/c^{2}')
    string_list.append(f'#sigma {sigma*1000:.2f} #pm {sigma_error*1000:.2f} MeV/c^{2}')

    if roo_data.sumEntries()>0:
        string_list.append('#chi^{2} / NDF ' + f'{frame.chiSquare(6 if bkg_model=="pol2" else 5):.2f}')

    # string_list.append(f'Significance ({nsigma:.0f}#sigma) {signif:.1f} #pm {signif_error:.1f}')
    # string_list.append(f'S ({nsigma:.0f}#sigma) {signal_counts} #pm {int(round(math.sqrt(signal_counts)))}')
    # string_list.append(f'B ({nsigma:.0f}#sigma) {background_counts} #pm {int(round(math.sqrt(signal_counts)))}')

    # if background_counts > 0:
    #     ratio = signal_counts / background_counts
    #     string_list.append(f'S/B ({nsigma:.0f}#sigma) {ratio:.2f}')

    for s in string_list:
        pinfo.AddText(s)

    frame.addObject(pinfo)

    sub_dir_name = f'pT{pt_range[0]}{pt_range[1]}_eff{eff:.2f}{split}'
    sub_dir = output_dir.GetDirectory(sub_dir_name)

    if not sub_dir:
        sub_dir = output_dir.mkdir(f'pT{pt_range[0]}{pt_range[1]}_eff{eff:.2f}{split}')

    sub_dir.cd()

    frame.Write(f'frame_model_{bkg_model}')
    hyp_mass.Write(f'hyp_mass_model{bkg_model}')
    width.Write(f'width_model{bkg_model}')



def estimate_maxima(kde):
    
    no_samples = 10000

    samples = np.linspace(2.990, 2.992, no_samples)
    probs = kde.evaluate(samples)
    maxima_index = probs.argmax()
    maxima = samples[maxima_index]

    return maxima
