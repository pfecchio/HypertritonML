#!/usr/bin/env python3
import argparse
import os
import time
import warnings
from array import array
import math

import numpy as np
import yaml

import hyp_analysis_utils as hau
import pandas as pd

import ROOT
from ROOT import TFile, gROOT
from analysis_classes import ModelApplication

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()
ROOT.ROOT.EnableImplicitMT()
ROOT.RooMsgService.instance().setSilentMode(True)

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('-u', '--unbinned', help='Perform unbinned fit', action='store_true')
parser.add_argument('config', help='Path to the YAML configuration file')
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
LARGE_DATA = params['LARGE_DATA']
LOAD_LARGE_DATA = params['LOAD_LARGE_DATA']

CENT_CLASSES = params['CENTRALITY_CLASS']
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']
COLUMNS = params['TRAINING_COLUMNS']

SPLIT_MODE = args.split
UNBINNED_FIT = args.unbinned
PASS = os.getenv('HYPERML_PASS')
OTF = os.getenv('HYPERML_OTF')

OTF = ''

SPLIT_CUTS = ['']
SPLIT_LIST = ['']
if SPLIT_MODE:
    SPLIT_LIST = ['_matter','_antimatter']
    SPLIT_CUTS = ['&& ArmenterosAlpha > 0', '&& ArmenterosAlpha < 0']

CT = False

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
bkg_path = os.path.expandvars(params['BKG_PATH'])
data_path = os.path.expandvars(params['DATA_PATH'])
analysis_res_path = os.path.expandvars(params['ANALYSIS_RESULTS_PATH'])

BKG_MODELS = params['BKG_MODELS']

results_dir = os.environ['HYPERML_RESULTS_{}'.format(N_BODY)]

###############################################################################
start_time = time.time()                          # for performances evaluation

file_name = results_dir + f'/{FILE_PREFIX}_std_results_pass{PASS}{OTF}.root'
results_file = TFile(file_name, 'recreate')

standard_selection = 'V0CosPA > 0.9999 && NpidClustersHe3 > 80 && He3ProngPt > 1.8 && pt > 2 && pt < 10 && PiProngPt > 0.15 && He3ProngPvDCA > 0.05 && PiProngPvDCA > 0.2 && TPCnSigmaHe3 < 3.5 && TPCnSigmaHe3 > -3.5 && ProngsDCA < 1'


rdfData = ROOT.RDataFrame('DataTable',data_path)
rdfMC = ROOT.RDataFrame('SignalTable', signal_path)

mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2.975, 3.01, 'GeV/c^{2}')

bLfunction = ROOT.TF1('bLfunction', '1115.683 + 1875.61294257 - [0]', 0, 10)

for split, splitcut in zip(SPLIT_LIST,SPLIT_CUTS):
    NBINS = len(CT_BINS) - 1 if CT  else len(PT_BINS) - 1
    BINS = np.asarray(CT_BINS if CT  else PT_BINS, dtype=np.float64) 
    hist_massesData = {}
    hist_massesMC = {}
    hist_delta_shift = {}
    for bkgmodel in BKG_MODELS:
        hist_massesData[bkgmodel] = ROOT.TH1D(f'massData_{split}_{bkgmodel}', ';#it{p}_{T} (GeV/#it{c});#it{c}t (cm);m (MeV/#it{c}^{2})', NBINS, BINS)
        hist_massesMC[bkgmodel] = ROOT.TH1D(f'massMC_{split}_{bkgmodel}', ';#it{p}_{T} (GeV/#it{c});#it{c}t (cm);m (MeV/#it{c}^{2})', NBINS, BINS)

    dfData_applied = rdfData.Filter(standard_selection + splitcut)
    dfMC_applied = rdfMC.Filter(standard_selection + splitcut)

    for cclass in CENT_CLASSES:
        cent_dir = results_file.mkdir(f'{cclass[0]}-{cclass[1]}{split}')
        dfData_cent = dfData_applied.Filter(f'centrality >= {cclass[0]} && centrality < {cclass[1]}')
        dfMC_cent = dfMC_applied.Filter(f'centrality >= {cclass[0]} && centrality < {cclass[1]}')
        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
            pt = 0.5 * (ptbin[0] + ptbin[1])
            for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                ct = 0.5 * (ctbin[0] + ctbin[1])
                binNo = hist_massesMC['expo'].GetXaxis().FindBin(ct) if CT  else hist_massesMC['expo'].GetXaxis().FindBin(pt)
                
                mass_bins = 35
                sub_dir = cent_dir.mkdir(f'ct_{ctbin[0]}{ctbin[1]}') if CT  else cent_dir.mkdir(f'pt_{ptbin[0]}{ptbin[1]}')
                sub_dir.cd()
                massData_array = dfData_cent.Filter(f'ct<{ctbin[1]} && ct>{ctbin[0]} && pt<{ptbin[1]} && pt>{ptbin[0]}').AsNumpy(['m'])
                massMC_array = dfMC_cent.Filter(f'ct<{ctbin[1]} && ct>{ctbin[0]} && pt<{ptbin[1]} && pt>{ptbin[0]}').AsNumpy(['m'])
                if UNBINNED_FIT:
                    roo_data = hau.ndarray2roo(massData_array['m'], mass)
                    roo_mc = hau.ndarray2roo(massMC_array['m'], mass)
                else:
                    countsData = np.histogram(massData_array['m'], mass_bins, range=[2.975, 3.01])
                    countsMC = np.histogram(massMC_array['m'], mass_bins, range=[2.975, 3.01])
                    h1_minvData = hau.h1_invmass(countsData, cclass, ptbin, ctbin, name='data')
                    h1_minvMC = hau.h1_invmass(countsMC, cclass, ptbin, ctbin, name='MC')
                    roo_data = ROOT.RooDataHist(f'Roo{h1_minvData.GetName()}', 'Data', ROOT.RooArgList(mass), h1_minvData)
                    roo_mc = ROOT.RooDataHist(f'Roo{h1_minvMC.GetName()}', 'MC', ROOT.RooArgList(mass), h1_minvMC)

                # define RooFit objects                
                mean_gen = ROOT.RooRealVar('mean_gen', 'hyp_mass_mc', 2.989, 2.993, 'GeV^-1')
                width_gen = ROOT.RooRealVar('width_gen', 'width_gen',0.0001, 0.005, 'GeV/c^2')
                width2_gen = ROOT.RooRealVar('width2_gen', 'width2_gen',0.0001, 0.01, 'GeV/c^2')
                
                # funtion for modelling and generating MC mass
                sig_gen = ROOT.RooGaussian('sig_gen', 'sig_gen', mass, mean_gen, width_gen)
                res_gen = ROOT.RooGaussian('res_gen', 'res_gen', mass, mean_gen, width2_gen)
                nsig_gen = ROOT.RooRealVar('nsig_gen', 'nsig_gen', 0., 1.)
                sigres_gen = ROOT.RooAddPdf('sigres_gen', 'signal + resolution', ROOT.RooArgList(sig_gen, res_gen),ROOT.RooArgList(nsig_gen))
                
                sigres_gen.fitTo(roo_mc)
                xframe = mass.frame(mass_bins)
                xframe.SetName(f'frameMC_ct{ctbin[0]}{ctbin[1]}_pT{ptbin[0]}{ptbin[1]}_cen{cclass[0]}{cclass[1]}')
                roo_mc.plotOn(xframe)
                sigres_gen.plotOn(xframe)
                sigres_gen.paramOn(xframe)

                for bkgmodel in BKG_MODELS:
                    # signal for data fit
                    mean = ROOT.RooRealVar('mean', 'hyp_mass', 2.989, 2.993, 'GeV^-1')
                    width = ROOT.RooRealVar('width', 'width', 0.0001, 0.005, 'GeV/c^2')
                    signal = ROOT.RooGaussian('signal', 'signal', mass, mean, width)

                    if bkgmodel == 'pol1':
                        c0 = ROOT.RooRealVar('c0','constant c0',-100., 100.,'GeV/c^{2}')
                        c1 = ROOT.RooRealVar('c1', 'constant c1', -100., 100., 'GeV/c^{2}')
                        background = ROOT.RooPolynomial('background', 'pol1 for bkg', mass, ROOT.RooArgList(c0, c1))
                        ndf = 5

                    if bkgmodel == 'pol2':
                        c0 = ROOT.RooRealVar('c0','constant c0',-100., 100.,'GeV/c^{2}')
                        c1 = ROOT.RooRealVar('c1', 'constant c1', -100., 100., 'GeV/c^{2}')
                        c2 = ROOT.RooRealVar('c2','constant c2',-100., 100.,'GeV/c^{2}')
                        background = ROOT.RooPolynomial('background', 'pol1 for bkg', mass, ROOT.RooArgList(c0, c1, c2))
                        ndf = 6

                    if bkgmodel == 'expo':
                        slope = ROOT.RooRealVar('slope', 'slope mass', -100., 100., 'GeV')
                        background = ROOT.RooExponential('background', 'expo for bkg', mass, slope)
                        ndf = 4

                    n = ROOT.RooRealVar('n', 'n', 0., 1.)
                    data_model = ROOT.RooAddPdf(f'data_model_{bkgmodel}', 'signal + background', ROOT.RooArgList(signal, background), ROOT.RooArgList(n))

                    data_model.fitTo(roo_data)

                    mu_meas = mean.getVal()*1000.    # in MeV
                    mu_meas_error = mean.getError()*1000.  # in MeV

                    sigma_meas = width.getVal()*1000.    # in MeV
                    sigma_meas_error = width.getError()*1000.    # in MeV

                    # compute significance
                    nsigma = 3
                    
                    mass.setRange('signal_region', (mu_meas - (nsigma * sigma_meas)) / 1000., (mu_meas + (nsigma * sigma_meas)) / 1000.)
                    mass_set = ROOT.RooArgSet(mass)
                    mass_norm_set = ROOT.RooFit.NormSet(mass_set)

                    frac_signal_range = signal.createIntegral(mass_set, mass_norm_set, ROOT.RooFit.Range('signal_region'))
                    frac_background_range = background.createIntegral(mass_set, mass_norm_set, ROOT.RooFit.Range('signal_region'))

                    signal_counts = int(roo_data.sumEntries() * n.getVal() * frac_signal_range.getVal())
                    background_counts = int(roo_data.sumEntries() * (1 - n.getVal()) * frac_background_range.getVal())

                    significance = signal_counts / math.sqrt(signal_counts + background_counts + 1e-10)
                    significance_error = hau.significance_error(signal_counts, background_counts)

                    frame1 = mass.frame(mass_bins)
                    roo_data.plotOn(frame1, ROOT.RooFit.Name('data'))
                    data_model.plotOn(frame1, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Name('model'))
                    data_model.plotOn(frame1, ROOT.RooFit.Components('signal'), ROOT.RooFit.LineColor(ROOT.kRed))
                    data_model.plotOn(frame1, ROOT.RooFit.Components('background'), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kBlue))

                    chi2_red = frame1.chiSquare('model', 'data', ndf)

                    pinfo = ROOT.TPaveText(0.558, 0.652, 0.936, 0.875, 'NDC')
                    pinfo.SetBorderSize(0)
                    pinfo.SetFillStyle(0)
                    pinfo.SetTextAlign(30+3)
                    pinfo.SetTextFont(42)

                    pinfo.AddText(f'#mu = {mu_meas:.3f} #pm {mu_meas_error:.3f} MeV/c^{2}')
                    pinfo.AddText(f'#sigma = {sigma_meas:.2f} #pm {sigma_meas_error:.2f} MeV/c^{2}')
                    pinfo.AddText('#chi^{2} = ' + f'{chi2_red:.1f}')
                    pinfo.AddText(f'S = {significance:.1f} #pm {significance_error:.1f}')

                    frame1.addObject(pinfo)
                    frame1.Write()

                    # resid_hist = frame1.residHist('data', 'model')
                    # pull_hist = frame1.pullHist('data', 'model')

                    # frame2 = mass.frame(ROOT.RooFit.Title('Residual Distribution'))
                    # frame2.addPlotable(resid_hist,'P')

                    # frame3 = mass.frame(ROOT.RooFit.Title('Pull Distribution'))
                    # frame3.addPlotable(pull_hist,'P')

                    dataHist = hist_massesData[bkgmodel]
                    dataHist.SetBinContent(binNo, mu_meas)
                    dataHist.SetBinError(binNo, mu_meas_error)

                    hist_massesMC[bkgmodel].SetBinContent(binNo, mean_gen.getVal()*1000 - 2991.31)
                    hist_massesMC[bkgmodel].SetBinError(binNo, mean_gen.getError()*1000)

                    # c = ROOT.TCanvas(f'{bkgmodel}_canvas', '', 1350, 450)
                    # c.Divide(3)

                    # c.cd(1)
                    # ROOT.gPad.SetLeftMargin(0.15)
                    # frame1.GetYaxis().SetTitleOffset(1.6)
                    # frame1.Draw()

                    # c.cd(2)
                    # ROOT.gPad.SetLeftMargin(0.15)
                    # frame2.GetYaxis().SetTitleOffset(1.6)
                    # frame2.Draw()

                    # c.cd(3)
                    # ROOT.gPad.SetLeftMargin(0.15)
                    # frame3.GetYaxis().SetTitleOffset(1.6)
                    # frame3.Draw()

                    # c.Write()

    results_file.cd()
    for bkgmodel in BKG_MODELS:
        h = hist_massesData[bkgmodel]
        mass_gauss = h.Clone()
        mass_gauss.Add(hist_massesMC[bkgmodel], -1)
        print(mass_gauss.GetName())
        mass_gauss.Fit(bLfunction)
        mass_gauss.Write(f'{h.GetName()}_gauss')
        h.Write()
        hist_massesMC[bkgmodel].Write()

print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')
