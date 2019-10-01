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

selected = df.Filter('V0CosPA > 0.9999 && NpidClustersHe3 > 80 && He3ProngPt > 1.8 && He3ProngPt < 10  && PiProngPt > 0.15 && He3ProngPvDCA > 0.05 && PiProngPvDCA > 0.2 && std::abs(TPCnSigmaHe3) < 3.5 && ProngsDCA < 1')
binning = np.array(params['CT_BINS'],dtype=np.float64)
nbins = len(binning) - 1
h2 = selected.Histo2D(("InvMass",";Invariant Mass;#it{ct} (cm);",36, 2.96, 3.05,nbins, binning),"InvMass","ct")
h2.Write()


results_file.Close()