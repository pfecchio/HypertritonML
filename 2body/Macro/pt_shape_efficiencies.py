import os

import numpy as np

import pandas as pd
import uproot
from ROOT import TH1D, gROOT, TFile
from root_numpy import fill_hist

gROOT.LoadMacro("../TreeToTables/GenerateTableFromMC.cc")
from ROOT import GenerateTableFromMC

input_dir = os.environ['HYPERML_DATA_2']+ "/splines_trees"
output_dir = os.environ['HYPERML_TABLES_2']+ "/splines_tables"

ct_bins = np.array([0,1,2,4,6,8,10,14,18,23,35],"d")


pt_shape_list = ["bol", "mtexp", "bw"]

hist_list = []

for pt_shape in pt_shape_list:
    th1_rec = TH1D("PreselEff_" + pt_shape, ";c#it{t} (cm);Preselection efficiency", len(ct_bins)-1, ct_bins)
    th1_sim = TH1D("", "", len(ct_bins)-1, ct_bins)
    GenerateTableFromMC(True, input_dir, output_dir, pt_shape)
    rec = uproot.open(output_dir+"/SignalTable.root")["SignalTable"].array("ct")
    sim = uproot.open(output_dir+"/SignalTable.root")["GenTable"].array("ct")
    fill_hist(th1_rec, rec)
    fill_hist(th1_sim, sim)
    th1_rec.Divide(th1_sim)
    hist_list.append(th1_rec)


tfile = TFile(os.environ['HYPERML_EFFICIENCIES_2'] + "/pt_shape_comparison", "recreate")
for hist in hist_list:
    hist.Write()
tfile.Close()

