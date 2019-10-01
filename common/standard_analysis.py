#!/usr/bin/env python3

import argparse
from ROOT import RDataFrame as RDF
from ROOT import TF1, TH1D, TH2D, TCanvas, TFile, TPaveText, gDirectory, gStyle, gROOT, TIter, TKey, TAxis
import analysis_utils as au
import ROOT
import os
import yaml
import numpy as np
import random

random.seed(1989)

parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to the YAML configuration file")
args = parser.parse_args()

ROOT.ROOT.EnableImplicitMT()
gROOT.SetBatch()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# define paths for loading data and storing results
mc_path = os.path.expandvars(params['MC_PATH'])
data_path = os.path.expandvars(params['DATA_PATH'])
dataDF = RDF('DataTable', data_path)
mcDF = RDF('SignalTable', mc_path)
genDF = RDF('GenTable', mc_path)

results_dir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]
file_name = results_dir + '/' + params['FILE_PREFIX'] + '_std.root'
results_file = TFile(file_name, 'recreate')


for cclass in params['CENTRALITY_CLASS']:
    cent_dir = results_file.mkdir('{}-{}'.format(cclass[0], cclass[1]))
    dataCentDF = dataDF.Filter(
        'centrality >= {} && centrality < {}'.format(cclass[0], cclass[1]))
    mcCentDF = mcDF.Filter(
        'centrality >= {} && centrality < {}'.format(cclass[0], cclass[1]))
    genCentDF = genDF.Filter(
        'centrality >= {} && centrality < {}'.format(cclass[0], cclass[1]))
    genSelected = genCentDF.Filter('std::abs(rapidity) < 0.5')

    bkg_models = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']
    fit_directories = []
    h2raw_counts = []

    ctBins = np.array(params['CT_BINS'], dtype=np.float64)
    nCtBins = len(ctBins) - 1
    ptBins = np.array(params['PT_BINS'], dtype=np.float64)
    nPtBins = len(ptBins) - 1
    nMBins = 36
    mBins = np.linspace(2.96, 3.05, nMBins + 1, dtype=np.float64)

    cosPAcuts = np.linspace(0.99985, 0.99995, 11)
    pidHe3cuts = np.arange(80, 101)
    he3PtCuts = np.linspace(1.7, 1.9, 3)
    piPtCuts = np.linspace(0.13, 0.18, 3)
    prongsDCA = np.linspace(0.8, 1.2, 5)

    rdfModel = ROOT.RDF.TH3DModel(
        "data", ";Invariant Mass;#it{p}_{T} (GeV/#it{c});#it{ct} (cm)", nMBins, mBins, nPtBins, ptBins, nCtBins, ctBins)
    rdfMCModel = ROOT.RDF.TH3DModel(
        "mc", ";Invariant Mass;#it{p}_{T} (GeV/#it{c});#it{ct} (cm)", nMBins, mBins, nPtBins, ptBins, nCtBins, ctBins)
    rdfGenModel = ROOT.RDF.TH2DModel(
        "gen", ";#it{p}_{T} (GeV/#it{c});#it{ct} (cm)", nPtBins, ptBins, nCtBins, ctBins)
    h2GenPtCt = genSelected.Histo2D(rdfGenModel, "pT", "ct")
    h1Tau = TH1D("systematics",";#tau (ps);Trials",150,150,300)
    recSelString = []
    for cosPA in cosPAcuts:
        for pidHe3 in pidHe3cuts:
            for he3Pt in he3PtCuts:
                for piPt in piPtCuts:
                    for pDCA in prongsDCA:
                        recSelString.append('V0CosPA > {} && NpidClustersHe3 > {} && He3ProngPt > {} && HypCandPt > 2 && HypCandPt < 10 && PiProngPt > {} && He3ProngPvDCA > 0.05 && PiProngPvDCA > 0.2 && std::abs(TPCnSigmaHe3) < 3.5 && ProngsDCA < {}'.format(
                            cosPA, pidHe3, he3Pt, piPt, pDCA))

    stringSample = random.sample(recSelString,50)
    stringSample.insert(0, 'V0CosPA > 0.9999 && NpidClustersHe3 > 80 && He3ProngPt > 1.8 && HypCandPt > 2 && HypCandPt < 10 && PiProngPt > 0.15 && He3ProngPvDCA > 0.05 && PiProngPvDCA > 0.2 && std::abs(TPCnSigmaHe3) < 3.5 && ProngsDCA < 1')
    savePlots = True
    for selString in stringSample:
        dataSelected = dataCentDF.Filter(selString)
        mcSelected = mcCentDF.Filter(selString)
        h3DataMassPtCt = dataSelected.Histo3D(
            rdfModel, "InvMass", "HypCandPt", "ct")
        h3McMassPtCt = mcSelected.Histo3D(
            rdfMCModel, "InvMass", "HypCandPt", "ct")
        h2RecCosPApt = mcCentDF.Histo2D(
            ("cosPA", ";#it{p}_{T} (GeV/#it{c}); cos#theta_{P};", 200, 0, 10, 2000, 0.98, 1), "HypCandPt", "V0CosPA")
        if savePlots:
            cent_dir.cd()
            h3DataMassPtCt.Write()
            h3McMassPtCt.Write()
            h2GenPtCt.Write()
            h2RecCosPApt.Write()

            for model in bkg_models:
                fit_directories.append(cent_dir.mkdir(model))
                h2raw_counts.append(TH2D('RawCounts_{}'.format(model), ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Raw counts',
                                         len(params['PT_BINS'])-1, np.array(
                                             params['PT_BINS'], 'double'), len(params['CT_BINS']) - 1,
                                         np.array(params['CT_BINS'], 'double')))

        for ptbin in zip(params['PT_BINS'][:-1], params['PT_BINS'][1:]):
            ptbin_index = h2raw_counts[0].GetXaxis().FindBin(
                0.5 * (ptbin[0] + ptbin[1]))

            for ctbin in zip(params['CT_BINS'][:-1], params['CT_BINS'][1:]):
                ctbin_index = h2raw_counts[0].GetYaxis().FindBin(
                    0.5 * (ctbin[0] + ctbin[1]))
                # binSelection = 'HypCandPt >= {} && HypCandPt < {} && ct >= {} && ct < {}'.format(
                #   ptbin[0],ptbin[1], ctbin[0], ctbin[1])
                # binDS = dataSelected.Filter(binSelection)
                # massVec = binDS.AsNumpy("InvMass")
                # unBinDataSet = ROOT.ROOT.UnBinData(massVec.size(), massVec.data())

                binName = "ct{}{}_pT{}{}_cen{}{}".format(
                    ctbin[0], ctbin[1], ptbin[0], ptbin[1], cclass[0], cclass[1])
                baseHisto = h3DataMassPtCt.ProjectionX(
                    binName, ptbin_index, ptbin_index, ctbin_index, ctbin_index)

                for model, fitdir, h2raw in zip(bkg_models, fit_directories, h2raw_counts):
                    histo = baseHisto.Clone(binName + "_{}".format(model))
                    fitdir = fitdir if savePlots else None
                    hyp_yield, err_yield = au.fitHist(
                        histo, ctbin, ptbin, cclass, fitdir, model=model)

                    h2raw.SetBinContent(ptbin_index, ctbin_index, hyp_yield)
                    h2raw.SetBinError(ptbin_index, ctbin_index, err_yield)

        # write on file
        cent_dir.cd()

        if savePlots:
            for h2raw in h2raw_counts:
                h2raw.Write()

        # Temporary for the ct spectra
        expo = TF1("myexpo", "[0]*exp(-x/[1]/0.029979245800)", 0, 28)
        expo.SetParLimits(1, 100, 350)
        h1GenCt = h2GenPtCt.ProjectionY("gen_ct")
        h1EffCt = h3McMassPtCt.ProjectionZ("eff_ct")
        h1EffCt.Divide(h1GenCt)
        for iBin in range(1, h1EffCt.GetNbinsX() + 1):
            h1EffCt.SetBinError(iBin, 0)

        for model, h2raw in zip(bkg_models, h2raw_counts):
            h1RawCt = h2raw.ProjectionY("ct_{}".format(model))
            h1RawCt.Divide(h1EffCt)
            h1RawCt.Scale(1, "width")
            h1RawCt.Fit(expo)
            h1Tau.Fill(expo.GetParameter(1))
            if savePlots:
                h1RawCt.Write()
        if savePlots:
            h1EffCt.Write()
            savePlots = False
    cent_dir.cd()
    h1Tau.Write()

results_file.Close()
