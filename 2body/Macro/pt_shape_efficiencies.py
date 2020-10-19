import os

import numpy as np

import pandas as pd
import uproot
from ROOT import TH1D, gROOT, TFile
import aghast

gROOT.LoadMacro("../TreeToTables/GenerateTableFromMC.cc")
from ROOT import GenerateTableFromMC

input_dir = os.environ['HYPERML_DATA_2']+ "/splines_trees"
output_dir = os.environ['HYPERML_TABLES_2']+ "/splines_tables"

ct_bins = np.array([0,1,2,4,6,8,10,14,18,23,35],"d")


pt_shape_list = ["bol", "mtexp", "bw"]

hist_list = []

for pt_shape in pt_shape_list:
    GenerateTableFromMC(True, input_dir, output_dir, pt_shape)
    rec = np.histogram(uproot.open(output_dir+"/SignalTable.root")["SignalTable"].array("ct"), bins=ct_bins)
    sim = np.histogram(uproot.open(output_dir+"/SignalTable.root")["GenTable"].array("ct"), bins=ct_bins)
    th1_rec = aghast.to_root(rec, "PreselEff_" + pt_shape)
    th1_rec.SetTitle(";c#it{t} (cm);Preselection efficiency")
    th1_sim = aghast.to_root(sim, "Sim_" + pt_shape)
    th1_rec.Divide(th1_sim)
    hist_list.append(th1_rec)


tfile = TFile(os.environ['HYPERML_EFFICIENCIES_2'] + "/pt_shape_comparison", "recreate")
for hist in hist_list:
    hist.Write()
tfile.Close()

