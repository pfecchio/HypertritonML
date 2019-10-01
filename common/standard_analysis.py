#!/usr/bin/env python3

from ROOT import RDataFrame as RDF
from ROOT import TF1, TH1D, TH2D, TCanvas, TFile, TPaveText, gDirectory, gStyle, gROOT, TIter, TKey, TAxis
import analysis_utils as au
import ROOT
import os
import yaml
import numpy as np

import argparse
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
results_dir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]
df = RDF('DataTable', data_path)

results_dir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]
file_name = results_dir + '/' + params['FILE_PREFIX'] + '_std.root'
results_file = TFile(file_name, 'recreate')


for cclass in params['CENTRALITY_CLASS']:
    cent_dir = results_file.mkdir('{}-{}'.format(cclass[0], cclass[1]))
    centDF = df.Filter('centrality >= {} && centrality < {}'.format(cclass[0], cclass[1]))

    h2seleff = TH2D('SelEff', ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Preselection efficiency', len(params['PT_BINS'])-1, np.array(
        params['PT_BINS'], 'double'), len(params['CT_BINS'])-1, np.array(params['CT_BINS'], 'double'))

    bkg_models = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']
    fit_directories = []
    h2raw_counts = []


    ctBins = np.array(params['CT_BINS'],dtype=np.float64)
    nCtBins = len(ctBins) - 1
    ptBins = np.array(params['PT_BINS'],dtype=np.float64)
    nPtBins = len(ptBins) - 1
    nMBins = 36
    mBins = np.linspace(2.96, 3.05, nMBins + 1,dtype=np.float64)
    selected = centDF.Filter('V0CosPA > 0.9999 && NpidClustersHe3 > 80 && He3ProngPt > 1.8 && HypCandPt > 2 && HypCandPt < 10 && PiProngPt > 0.15 && He3ProngPvDCA > 0.05 && PiProngPvDCA > 0.2 && std::abs(TPCnSigmaHe3) < 3.5 && ProngsDCA < 1')
    rdfModel = ROOT.RDF.TH3DModel("InvMass",";Invariant Mass;#it{ct} (cm);#it{p}_{T} (GeV/#it{c})",nMBins, mBins, nCtBins, ctBins, nPtBins, ptBins)
    h3MassPtCt = selected.Histo3D(rdfModel,"InvMass","ct","HypCandPt")

    h3MassPtCt.Write()
    for model in bkg_models:
        fit_directories.append(cent_dir.mkdir(model))
        h2raw_counts.append(TH2D('RawCounts_{}'.format(model), ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Raw counts',
                                len(params['PT_BINS'])-1, np.array(params['PT_BINS'], 'double'), len(params['CT_BINS']) - 1,
                                np.array(params['CT_BINS'], 'double')))

    for ptbin in zip(params['PT_BINS'][:-1], params['PT_BINS'][1:]):
        ptbin_index = h2raw_counts[0].GetXaxis().FindBin(0.5 * (ptbin[0] + ptbin[1]))

        for ctbin in zip(params['CT_BINS'][:-1], params['CT_BINS'][1:]):
            ctbin_index = h2raw_counts[0].GetYaxis().FindBin(0.5 * (ctbin[0] + ctbin[1]))

            binName = "ct{}{}_pT{}{}_cen{}{}".format(
              ctbin[0],ctbin[1], ptbin[0], ptbin[1], cclass[0], cclass[1])
            baseHisto = h3MassPtCt.ProjectionX(binName, ctbin_index, ctbin_index, ptbin_index, ptbin_index)
            
            for model, fitdir, h2raw in zip(bkg_models, fit_directories, h2raw_counts):
                histo = baseHisto.Clone(binName + "_{}".format(model))
                hyp_yield, err_yield = au.fitHist(histo, ctbin, ptbin, cclass, fitdir, model=model)

                h2raw.SetBinContent(ptbin_index, ctbin_index, hyp_yield)
                h2raw.SetBinError(ptbin_index, ctbin_index, err_yield)

    # write on file
    cent_dir.cd()
    h2seleff.Write()

    for h2raw in h2raw_counts:
        h2raw.Write()

results_file.Close()