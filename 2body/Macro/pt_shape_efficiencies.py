import os

import numpy as np

import pandas as pd
import uproot
from ROOT import TH1D, gROOT, TFile
import aghast

gROOT.LoadMacro('../TreeToTables/GenerateTableFromMC.cc')
from ROOT import GenerateTableFromMC

input_dir = os.environ['HYPERML_TREES__2']
output_dir = os.environ['HYPERML_TABLES_2']

ct_bins = np.array([0,1,2,4,6,8,10,14,18,23,35], 'd')


pt_shape_list = ['bol', 'mtexp', 'bw']

hist_list = []

for pt_shape in pt_shape_list:
    GenerateTableFromMC(True, input_dir, output_dir, pt_shape)

    numpy_rec = np.histogram(uproot.open(output_dir + '/SignalTable_20g7.root')['SignalTable'].array('ct'), bins=ct_bins)
    numpy_sim = np.histogram(uproot.open(output_dir + '/SignalTable_20g7.root')['GenTable'].array('ct'), bins=ct_bins)

    ghastly_rec = aghast.from_numpy(numpy_rec)
    ghastly_sim = aghast.from_numpy(numpy_sim)

    th1_rec = aghast.to_root(ghastly_rec, 'PreselEff_' + pt_shape)
    th1_rec.SetTitle(';c#it{t} (cm);Preselection efficiency')

    th1_sim = aghast.to_root(ghastly_sim, 'Sim_' + pt_shape)

    th1_rec.Divide(th1_sim)
    hist_list.append(th1_rec)


tfile = TFile(os.environ['HYPERML_UTILS_2'] + '/pt_shape_comparison.root', 'recreate')

for hist in hist_list:
    hist.Write()
    
tfile.Close()

