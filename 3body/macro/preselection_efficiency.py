#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import yaml

import hyp_analysis_utils as hau
import pandas as pd
import uproot
from ROOT import TFile
import numpy as np


###############################################################################
CENT_CLASSES = [[0, 90]]
PT_BINS = [0,10]
CT_BINS = [2,4,6,8,10,14,18,23,35]

###############################################################################
# define paths for loading data and storing results
mc_table = Path(os.environ['HYPERML_TABLES_3'] + '/O2/SignalTableReweight.root')
results_file = Path(os.environ['HYPERML_EFFICIENCIES_3'] + f'/PreselEff_cent{CENT_CLASSES[0][0]}{CENT_CLASSES[0][1]}.root')

# open the table with uproot
df = uproot.open(mc_table)['SignalTable'].pandas.df().query('bw_accept')

df_den = df
df_num = df.query('cos_pa>0 and pt>2')

# define histograms for reconstructed and generated hypertritons
presel_histo = hau.h2_preselection_efficiency(PT_BINS, CT_BINS)
gen_histo = hau.h2_generated(PT_BINS, CT_BINS)

# fill the histograms
for pt, ct in np.asarray(df_num[['pt', 'ct']], dtype=np.double):
  presel_histo.Fill(pt, ct)
for pt, ct in np.asarray(df_den[['gPt', 'gCt']], dtype=np.double):
  gen_histo.Fill(pt, ct)

# compute the efficiency as a function of pt and ct (TH2)
presel_histo.Divide(gen_histo)

# rename the TH2
presel_histo.SetName('PreselEff')

# get the eff vs pt
eff_pt = presel_histo.ProjectionX()
eff_pt.SetDirectory(0)
eff_pt.SetName('PreselEff_pt')
# get the eff vs ct
eff_ct = presel_histo.ProjectionY()
eff_ct.SetDirectory(0)
eff_ct.SetName('PreselEff_ct')

###############################################################################
t_file = TFile(f'{results_file}', "recreate")

presel_histo.Write()
eff_pt.Write()
eff_ct.Write()

t_file.Close()