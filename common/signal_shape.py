#!/usr/bin/env python3

import argparse
import os
import time

import hyp_analysis_utils as hau
import hyp_plot_utils as hpu
import numpy as np
import pandas as pd
import ROOT
import yaml

import math

ROOT.gROOT.SetBatch()

np.random.seed(42)

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
###############################################################################

###############################################################################
# define some globals
FILE_PREFIX = params['FILE_PREFIX']

DATA_PATH = os.path.expandvars(params['DATA_PATH'])

SPLIT_LIST = ['_matter','_antimatter'] if False else ['']
BKG_MODELS = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']

CENT_CLASS = params['CENTRALITY_CLASS'][0]
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']

###############################################################################

###############################################################################
# input/output files
results_dir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]
tables_dir = os.path.dirname(DATA_PATH)
efficiency_dir = os.environ['HYPERML_EFFICIENCIES_{}'.format(params['NBODY'])]

# mc file
file_name = tables_dir + f'/applied_mc_df_{FILE_PREFIX}.parquet.gzip'
mc_df = pd.read_parquet(file_name, engine='fastparquet')

# significance scan output
file_name = results_dir + f'/Efficiencies/{FILE_PREFIX}_sigscan.npy'
sigscan_dict = np.load(file_name, allow_pickle=True).item()

# output file
file_name = results_dir + f'/{FILE_PREFIX}_signal_shape.root'
output_file = ROOT.TFile(file_name, 'recreate')
###############################################################################

###############################################################################
start_time = time.time()
###############################################################################

###############################################################################
# define support globals and methods for getting hypertriton counts
MC_MASS = 2.99131

def get_eff_index(eff):
    idx = (eff - EFF_MIN + EFF_STEP) * 100
    if isinstance(eff, np.ndarray):
        return idx.astype(int)

    return int(idx)


def get_effscore_dict(ctbin):
    info_string = f'090_210_{ctbin[0]}{ctbin[1]}'
    file_name = efficiency_dir + f'/Eff_Score_{info_string}.npy'

    return {round(e[0], 2): e[1] for e in np.load(file_name).T}

###############################################################################
# iterate over best efficiencies
eff_best_array = [round(sigscan_dict[f'ct{ctbin[0]}{ctbin[1]}pt210'][0], 2) for ctbin in zip(CT_BINS[:-1], CT_BINS[1:])]
eff_best_it = iter(eff_best_array)

# actual analysis
for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
    ct_dir = output_file.mkdir(f'ct{ctbin[0]}{ctbin[1]}')
    ct_dir.cd()

    score_dict = get_effscore_dict(ctbin)
    # get the data slice as a RooDataSet
    eff_best = next(eff_best_it)
    tsd = score_dict[eff_best]

    # define global RooFit objects
    mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2.971, 3.011, 'GeV/c^{2}')
    mass.setVal(MC_MASS)
    delta_mass = ROOT.RooRealVar('delta_m', '#Delta m', -0.0005, 0.0005, 'GeV/c^{2}')
    shift_mass = ROOT.RooAddition('shift_m', "m + #Delta m", ROOT.RooArgList(mass, delta_mass))

        # get mc slice for this ct bin
    mc_slice = mc_df.query('@ctbin[0]<ct<@ctbin[1] and 2.960<m<3.040')
    mc_array = np.array(mc_slice.query('score>@tsd')['m'].values, dtype=np.float64)
    np.random.shuffle(mc_array)
    roo_mc_slice = hau.ndarray2roo(mc_array if len(mc_array) < 5e4 else mc_array[:50000], mass)
    # roo_mc_slice = hau.ndarray2roo( mc_array[:1000], mass)

    signal_kde = ROOT.RooKeysPdf('signal_kde', 'signal kde', shift_mass, mass, roo_mc_slice, ROOT.RooKeysPdf.NoMirror, 2.)

    mu_gauss = ROOT.RooRealVar('mu_gauss', 'mu gauss', 2.989, 2.993, 'GeV/c^{2}')
    sigma_gauss = ROOT.RooRealVar('sigma_gauss', 'sigma gauss', 0.0001, 0.01, 'GeV/c^{2}')

    signal_gauss = ROOT.RooGaussian('signal_gauss', 'signal component pdf', mass, mu_gauss, sigma_gauss)

    fit_results_gauss = signal_gauss.fitTo(roo_mc_slice, ROOT.RooFit.Range(2.971, 3.011), ROOT.RooFit.NumCPU(32), ROOT.RooFit.Save())
    fit_results_kde = signal_kde.fitTo(roo_mc_slice, ROOT.RooFit.Range(2.971, 3.011), ROOT.RooFit.NumCPU(32), ROOT.RooFit.Save())

    canvas = ROOT.TCanvas(f'canvas_{eff_best}')
    canvas.SetLogy()

    xframe = mass.frame(160)
    xframe.SetName(f'mc_eff{eff_best:.2f}')

    roo_mc_slice.plotOn(xframe, ROOT.RooFit.Name('MC data'))
    signal_kde.plotOn(xframe, ROOT.RooFit.Name('KDE fit'), ROOT.RooFit.LineColor(ROOT.kBlue))
    signal_gauss.plotOn(xframe, ROOT.RooFit.Name('Gauss fit'), ROOT.RooFit.LineColor(ROOT.kRed))

    xframe.SetMinimum(1)
    xframe.SetMaximum(1e5)

    pinfo = ROOT.TPaveText(0.072, 0.746, 0.492, 0.845, 'NDC')
    pinfo.SetBorderSize(0)
    pinfo.SetFillStyle(0)
    pinfo.SetTextAlign(22)
    pinfo.SetTextFont(42)

    pinfo.AddText(f'({ctbin[0]} < ct < {ctbin[1]}) cm ')
    pinfo.AddText(f'BDT efficiency = {eff_best:.2f}')

    leg = ROOT.TLegend(0.67, 0.72, 0.99, 0.87)
    # leg.SetFillColor(ROOT.kWhite)
    # leg.SetLineColor(ROOT.kWhite)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(xframe.findObject('MC data'), 'MC data', 'P')
    leg.AddEntry(xframe.findObject('KDE fit'), 'KDE fit', 'L')
    leg.AddEntry(xframe.findObject('Gauss fit'), 'Gauss fit', 'L')

    xframe.addObject(pinfo)
    xframe.addObject(leg)
    xframe.Write()

    xframe.Draw()
    leg.Draw('same')

    canvas.Write()
 