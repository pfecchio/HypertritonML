#!/usr/bin/env python3

import argparse
import math
import os
import time

import hyp_analysis_utils as hau
import hyp_plot_utils as hpu
import numpy as np
import pandas as pd
import ROOT
import yaml

ROOT.gSystem.Load('RooCustomPdfs/libRooDSCBShape.so')
from ROOT import RooDSCBShape

ROOT.gROOT.SetBatch()
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)

kPurpleC = ROOT.TColor.GetColor('#911eb4')

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


def ratio_hist(roo_data, roo_pdf, roo_var, name, limits):
    params = roo_pdf.getParameters(ROOT.RooArgSet(roo_var))
    tf1_pdf = roo_pdf.asTF(ROOT.RooArgList(roo_var), ROOT.RooArgList(params), ROOT.RooArgSet(roo_var))
    
    h = roo_mc_slice.createHistogram('h_MC', roo_var, 195)

    h_ratio = h.Clone('h_ratio_' + name)
    h_ratio.SetTitle('')
    h_ratio.GetYaxis().SetTitle('(data - fit)/ fit  ')
    h_ratio.SetMarkerColor(kPurpleC)
    h_ratio.SetLineColor(kPurpleC)
    bin_width = h.GetBinWidth(1)

    leftmost_bin = h_ratio.FindBin(limits[0] + bin_width/2)
    rightmost_bin = h_ratio.FindBin(limits[1] - bin_width/2)

    for ibin in range(1, h_ratio.GetNbinsX() + 1):
        bin_center = h.GetBinCenter(ibin)
        if bin_center < limits[0] or bin_center > limits[1]:
            continue

        data_val = h.GetBinContent(ibin)
        data_err = h.GetBinError(ibin)
        fit_val = tf1_pdf.Eval(bin_center) * h.Integral(leftmost_bin, rightmost_bin) * bin_width

        ratio_val = (data_val - fit_val) / fit_val
        ratio_err = data_err / fit_val

        h_ratio.SetBinContent(ibin, ratio_val)
        h_ratio.SetBinError(ibin, ratio_err)
    
    h_ratio.GetXaxis().SetRangeUser(limits[0], limits[1])

    return h_ratio


def ratio_plot(canvas, frame, ratio_hist, limits):
    canvas.cd()

    pad_top = ROOT.TPad('pad_top', 'pad_top', 0.0, 0.4, 1.0, 1.0, 0)
    pad_top.SetLeftMargin(0.15)
    pad_top.SetBottomMargin(0.02)
    pad_top.SetLogy()
    pad_top.Draw()
    pad_top.cd()

    frame.GetYaxis().SetMaxDigits(3)
    frame.GetYaxis().SetRangeUser(8e1, 5e5)
    frame.GetXaxis().SetLabelOffset(1)
    frame.GetYaxis().SetTitleSize(22)
    frame.GetYaxis().SetTitleOffset(1.5)
    frame.GetXaxis().SetRangeUser(limits[0], limits[1])
    frame.Draw()

    canvas.cd()
    pad_bottom = ROOT.TPad('pad_bottom', 'pad_bottom', 0.0, 0.0, 1.0, 0.4, 0)
    pad_bottom.SetLeftMargin(0.15)
    pad_bottom.SetTopMargin(0.05)
    pad_bottom.SetBottomMargin(0.3)
    pad_bottom.Draw()
    pad_bottom.cd()

    ratio_hist.GetYaxis().SetRangeUser(-0.15, 0.15)
    ratio_hist.GetYaxis().SetNdivisions(505)
    # ratio_hist.GetYaxis().SetLabelSize(0.08)
    ratio_hist.GetYaxis().SetTitleSize(20)
    ratio_hist.GetYaxis().SetTitleOffset(1.5)
    # ratio_hist.GetXaxis().SetLabelSize(0.1)
    # ratio_hist.GetXaxis().SetDecimals()
    # ratio_hist.GetXaxis().SetTitleSize(0.1)
    ratio_hist.GetXaxis().SetTitleSize(22)
    ratio_hist.GetXaxis().SetTitleOffset(2.5)
    ratio_hist.GetXaxis().SetTickLength(0.05)
    ratio_hist.Draw('pe same')

    upper_line = ROOT.TLine(limits[0], 0.1, limits[1], 0.1)
    upper_line.SetLineStyle(ROOT.kDashed)
    upper_line.Draw('same')
    
    center_line = ROOT.TLine(limits[0], 0., limits[1], 0.)
    center_line.SetLineStyle(ROOT.kDashed)
    center_line.Draw('same')
    
    lower_line = ROOT.TLine(limits[0], -0.1,limits[1], -0.1)
    lower_line.SetLineStyle(ROOT.kDashed)
    lower_line.Draw('same')



###############################################################################
# iterate over best efficiencies
eff_best_array = [round(sigscan_dict[f'ct{ctbin[0]}{ctbin[1]}pt210'][0], 2) for ctbin in zip(CT_BINS[:-1], CT_BINS[1:])]
eff_best_it = iter(eff_best_array)

KDE_SIZE = [25, 50, 75, 100, 125, 150, 175, 200]
N_KDE = [6, 8, 7, 5, 4, 6, 4, 2, 1]

N_KDE_IT = iter(N_KDE)

results = {}

# actual analysis
for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
    ct_dir = output_file.mkdir(f'ct{ctbin[0]}{ctbin[1]}')
    ct_dir.cd()

    score_dict = get_effscore_dict(ctbin)
    # get the data slice as a RooDataSet
    eff_best = next(eff_best_it)
    tsd = score_dict[eff_best]

    # define global RooFit objects
    mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2.973, 3.012, 'GeV/c^{2}')
    mass.setVal(MC_MASS)
    delta_mass = ROOT.RooRealVar('delta_m', '#Delta m', -0.0005, 0.0005, 'GeV/c^{2}')
    shift_mass = ROOT.RooAddition('shift_m', "m + #Delta m", ROOT.RooArgList(mass, delta_mass))

    # get mc slice for this ct bin
    mc_slice = mc_df.query('@ctbin[0]<ct<@ctbin[1] and 2.973<m<3.012')
    mc_array = np.array(mc_slice.query('score>@tsd')['m'].values, dtype=np.float64)
    np.random.shuffle(mc_array)
    roo_mc_slice_kde = hau.ndarray2roo(mc_array if len(mc_array) < 5e4 else mc_array[:50000], mass)
    roo_mc_slice = hau.ndarray2roo(mc_array, mass)

    mu = ROOT.RooRealVar('mu', 'hypertriton mass', 2.989, 2.993, 'GeV/c^{2}')
    sigma = ROOT.RooRealVar('sigma', 'hypertriton width', 0.0001, 0.004, 'GeV/c^{2}') \
                
    a1 = ROOT.RooRealVar('a1', 'a1', 0, 5.)
    a2 = ROOT.RooRealVar('a2', 'a2', 0, 10.)
    n1 = ROOT.RooRealVar('n1', 'n1', 1, 10.)
    n2 = ROOT.RooRealVar('n2', 'n2', 1, 10.)

    signal_dscb = ROOT.RooDSCBShape('cb', 'cb', mass, mu, sigma, a1, n1, a2, n2)
    signal_kde = ROOT.RooKeysPdf('signal', 'signal', shift_mass, mass, roo_mc_slice_kde, ROOT.RooKeysPdf.NoMirror, 2.)

    fit_results_dscb = signal_dscb.fitTo(roo_mc_slice, ROOT.RooFit.Range(2.973, 3.012), ROOT.RooFit.NumCPU(32), ROOT.RooFit.Save())

    frame = mass.frame(195)
    roo_mc_slice.plotOn(frame, ROOT.RooFit.Name('MC'))
    signal_dscb.plotOn(frame, ROOT.RooFit.Name('DSCB pdf'), ROOT.RooFit.LineColor(ROOT.kGreen+1))
    signal_kde.plotOn(frame, ROOT.RooFit.Name('KDE pdf'), ROOT.RooFit.LineColor(ROOT.kBlue))

    pinfo = ROOT.TPaveText(0.066, 0.710, 0.486, 0.809, 'NDC')
    pinfo.SetBorderSize(0)
    pinfo.SetFillStyle(0)
    pinfo.SetTextAlign(22)
    pinfo.SetTextFont(42)

    pinfo.AddText(f'({ctbin[0]} < ct < {ctbin[1]}) cm ')
    pinfo.AddText(f'BDT efficiency = {eff_best:.2f}')

    leg = ROOT.TLegend(0.594, 0.458, 0.915, 0.641)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(frame.findObject('MC'), 'MC', 'PE')
    leg.AddEntry(frame.findObject('DSCB pdf'), 'DSCB pdf', 'L')
    leg.AddEntry(frame.findObject('KDE pdf'), 'KDE pdf', 'L')

    frame.addObject(pinfo)
    frame.addObject(leg)

    frame.Write()

    pinfo2 = ROOT.TPaveText(0.135, 0.67, 0.554, 0.813, 'NDC')
    pinfo2.SetBorderSize(0)
    pinfo2.SetFillStyle(0)
    pinfo2.SetTextAlign(22)
    pinfo2.SetTextFont(42)

    pinfo2.AddText(f'({ctbin[0]} < ct < {ctbin[1]}) cm ')
    pinfo2.AddText(f'BDT efficiency = {eff_best:.2f}')

    frame_kde = mass.frame(195)
    roo_mc_slice.plotOn(frame_kde, ROOT.RooFit.Name('MC data'))
    signal_kde.plotOn(frame_kde, ROOT.RooFit.Name('KDE pdf'), ROOT.RooFit.LineColor(ROOT.kBlue))

    leg_kde = ROOT.TLegend(0.655, 0.665, 0.976, 0.82)
    leg_kde.SetBorderSize(0)
    leg_kde.SetFillStyle(0)
    leg_kde.AddEntry(frame_kde.findObject('MC'), 'MC data', 'PE')
    leg_kde.AddEntry(frame_kde.findObject('KDE pdf'), 'KDE pdf', 'L')

    frame_kde.addObject(pinfo)
    frame_kde.addObject(leg_kde)

    frame_dscb = mass.frame(195)
    roo_mc_slice.plotOn(frame_dscb, ROOT.RooFit.Name('MC data'))
    signal_dscb.plotOn(frame_dscb, ROOT.RooFit.Name('DSCB pdf'), ROOT.RooFit.LineColor(ROOT.kBlue))

    leg_dscb = ROOT.TLegend(0.655, 0.665, 0.976, 0.82)
    leg_dscb.SetBorderSize(0)
    leg_dscb.SetFillStyle(0)
    leg_dscb.AddEntry(frame_dscb.findObject('MC'), 'MC data', 'PE')
    leg_dscb.AddEntry(frame_dscb.findObject('DSCB pdf'), 'DSCB pdf', 'L')

    frame_dscb.addObject(pinfo)
    frame_dscb.addObject(leg_dscb)


    h_ratio_kde = ratio_hist(roo_mc_slice, signal_kde, mass, 'kde', [2.985, 2.997])
    h_ratio_dscb = ratio_hist(roo_mc_slice, signal_dscb, mass, 'dscb', [2.985, 2.997])

    canvas_ratio_kde = ROOT.TCanvas('c_ratio_kde', '', 550, 525)
    ratio_plot(canvas_ratio_kde, frame_kde, h_ratio_kde, [2.985, 2.997])

    h_ratio_kde.Write()
    canvas_ratio_kde.Write()

    canvas_ratio_dscb = ROOT.TCanvas('c_ratio_dscb', '', 550, 525)
    ratio_plot(canvas_ratio_dscb, frame_dscb, h_ratio_dscb, [2.985, 2.997])

    h_ratio_dscb.Write()
    canvas_ratio_dscb.Write()





    # h_mc = roo_mc_slice.createHistogram('h_MC', mass, 500)
    # if (h_mc.GetSumw2N() == 0):
    #     h_mc.Sumw2(ROOT.kTRUE)
    # h_mc.Scale(1. / h_mc.GetEntries())
    # h_mc.SetLineColor(ROOT.kBlue)

    # roo_data_dscb = signal_dscb.generateBinned(mass, 10000000, ROOT.kTRUE)
    # h_dscb = roo_data_dscb.createHistogram('h_dscb', mass, 500)
    # if (h_dscb.GetSumw2N() == 0):
    #     h_dscb.Sumw2(ROOT.kTRUE)
    # h_dscb.Scale(1. / h_dscb.Integral())
    # h_dscb.SetLineColor(ROOT.kRed)
    
    # roo_data_kde = signal_kde.generateBinned(mass, 10000000, ROOT.kTRUE)
    # h_kde = roo_data_kde.createHistogram('h_kde', mass, 500)
    # if (h_kde.GetSumw2N() == 0):
    #     h_kde.Sumw2(ROOT.kTRUE)
    # h_kde.Scale(1. / h_kde.Integral())
    # h_kde.SetLineColor(ROOT.kRed)
    

    # ROOT.gStyle.SetOptStat(0)
    # c_dscb = ROOT.TCanvas('c_dscb', '')
    # c_dscb.SetTicks(0, 1)

    # rp_dscb = ROOT.TRatioPlot(h_mc, h_dscb)
    # rp_dscb.Draw()
    # rp_dscb.GetLowYaxis().SetNdivisions(505)

    # c_dscb.Write()

    # c_kde = ROOT.TCanvas('c_kde', '')
    # c_kde.SetTicks(0, 1)

    # rp_kde = ROOT.TRatioPlot(h_mc, h_kde)
    # rp_kde.Draw()
    # rp_kde.GetLowYaxis().SetNdivisions(505)

    # c_kde.Write()

    # for n in range(n_kde):
    #     kde_size = next(KDE_SIZE_IT)
    #     roo_mc_slice = hau.ndarray2roo(mc_array[:kde_size * 1000], mass)

    #     signal_kde = ROOT.RooKeysPdf('signal_kde', 'signal kde', shift_mass, mass, roo_mc_slice, ROOT.RooKeysPdf.NoMirror, 1.5)
    #     fit_results_kde = signal_kde.fitTo(roo_mc_slice, ROOT.RooFit.Range(2.960, 3.040), ROOT.RooFit.NumCPU(32), ROOT.RooFit.Save())

    #     res.append([delta_mass.getVal() * 1e6, delta_mass.getError() * 1e6])
        
    # results[f'{ctbin[0]}{ctbin[1]}'] = res

# print(results)
