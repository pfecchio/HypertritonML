#!/usr/bin/env python3
import argparse
import os
import time
import warnings

import numpy as np
import yaml

import hyp_analysis_utils as hau
import pandas as pd
import uproot
from ROOT import TFile, gROOT
from root_numpy import fill_hist

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
# def preselection_efficiency(df_reconstructed, df_generated, cent_class, ct_bins, pt_bins, save=True):
#     cent_cut = f'{cent_class[0]}<=centrality<={cent_class[1]}'

#     pres_histo = hau.h2_preselection_efficiency(pt_bins, ct_bins)
#     gen_histo = hau.h2_generated(pt_bins, ct_bins)

#     fill_hist(pres_histo, df_reconstructed.query(cent_cut)[['pt', 'ct']])
#     fill_hist(gen_histo, df_generated.query(cent_cut)[['gPt', 'gCt']])

#     pres_histo.Divide(gen_histo)

#     path = os.environ['HYPERML_EFFICIENCIES_3']

#     filename = path + f'/test_PreselEff_cent{cent_class[0]}{cent_class[1]}.root'
#     t_file = TFile(filename, "recreate")
                   
#     pres_histo.Write()
#     t_file.Close()


###############################################################################
CENT_CLASSES = [[0, 90]]
PT_BINS = [2, 10]
CT_BINS = [0,2,4,6,8,10,14,18,23]

###############################################################################
# define paths for loading data
# mc_file = os.path.expandvars(params['MC_PATH'])
# results_dir = os.environ['HYPERML_RESULTS_{}'.format(N_BODY)]

###############################################################################
# df_reco = df_signal.query('hasTOF_de and not bw_reject and not 1.1125<mppi_vert<1.1185')
# df_reco = df_signal.query('not bw_reject')

df_kf = uproot.open('~/run_nitty/mc/20200519_1838_covKf/HyperTritonTree3.root')['Hyp3O2'].pandas.df()
df_o2 = uproot.open('~/run_nitty/mc/20200519_1829_covO2/HyperTritonTree3.root')['Hyp3O2'].pandas.df()


pres_histo_kf = hau.h2_preselection_efficiency(PT_BINS, CT_BINS)
gen_histo_kf = hau.h2_generated(PT_BINS, CT_BINS)

pres_histo_o2 = hau.h2_preselection_efficiency(PT_BINS, CT_BINS)
gen_histo_o2 = hau.h2_generated(PT_BINS, CT_BINS)


fill_hist(pres_histo_kf, df_kf.query('pt>0')[['pt', 'ct']])
fill_hist(gen_histo_kf, df_kf[['gPt', 'gCt']])

fill_hist(pres_histo_o2, df_o2.query('pt>0')[['pt', 'ct']])
fill_hist(gen_histo_o2, df_o2[['gPt', 'gCt']])

pres_histo_kf.Divide(gen_histo_kf)
pres_histo_o2.Divide(gen_histo_o2)


    # path = os.environ['HYPERML_EFFICIENCIES_3']

    # filename = path + f'/test_PreselEff_cent{cent_class[0]}{cent_class[1]}.root'
t_file = TFile('eff.root', "recreate")

pres_histo_kf.SetName('PreselEff_KF')
pres_histo_o2.SetName('PreselEff_O2')
                   
pres_histo_kf.Write()
pres_histo_o2.Write()

t_file.Close()

# for cclass in CENT_CLASSES:
#     for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
#         for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
#             bin_selections = f'{cclass[0]}<=centrality<={cclass[1]} and {ctbin[0]}<gCt<{ctbin[1]} and {ptbin[0]}<gPt<{ptbin[1]}'

#             df_den = df_o2.query(bin_selections)
#             df_num = df_den.query(f'{ctbin[0]}<ct<{ctbin[1]}')

#             eff = len(df_num) / len(df_den)

#             print (eff)
            
#             del df_den, df_num

            