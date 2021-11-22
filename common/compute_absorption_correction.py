#!/usr/bin/env python3
import os
import re
import pickle
import warnings
import numpy as np
import ROOT
import yaml
import argparse

HE_3_MASS = 2.809230089
ct_bins = np.array([1, 2, 4, 6, 8, 10, 14, 18, 23, 35], dtype=float)
cent_bins = [0, 90]
##################################################################

if not os.path.isdir('../Results/2Body/absorption_correction'):
    os.mkdir('../Results/2Body/absorption_correction')

split_list = ['antimatter', 'matter']

# mc input file
mc_file = '/data/fmazzasc/PbPb_2body/absorption_studies/AnalysisResults_nominal.root'
outfile = ROOT.TFile("../Results/2Body/He3_abs_nominal.root", "recreate")

##################################################################
# functions
func = {}
func_max = {}

func_names = ["BGBW", "Boltzmann", "Mt-exp", "Pt-exp", "LevyTsallis"]

# functions input files
input_func_file = ROOT.TFile("../Utils/Anti_fits.root")

# get functions and maxima from file
for i_fun, _ in enumerate(func_names):
    key = f"cent_{cent_bins[0]}_{cent_bins[1]}_func_{func_names[i_fun]}"
    func[key] = input_func_file.Get(
        f"{func_names[i_fun]}/{0}/{func_names[i_fun]}{0}")
    func_max[key] = func[key].GetMaximum()

# book histograms
h_abs_radius = {}
h_abs_ct = {}
h_gen_radius = {}
h_gen_ct = {}
h_rec_radius = {}
h_rec_ct = {}

for key in func.keys():
    for split in (split_list + ['all']):
        h_abs_radius[f"{split}_" + key] = ROOT.TH1D(
            f"fAbsRadius_{split}_{cent_bins[0]}_" + key, ";#it{R}_{#it{abs}} (cm);Entries", 1000, 0, 1000)
        h_abs_ct[f"{split}_" + key] = ROOT.TH1D(
            f"fAbsCt_{split}_" + key, ";#it{c}t (cm);Entries", 2000, 1, 1000)
        h_gen_radius[f"{split}_" + key] = ROOT.TH1D(
            f"fGenRadius_{split}_" + key, ";#it{R}_{#it{abs}} (cm);Entries", 1000, 0, 1000)
        h_gen_ct[f"{split}_" + key] = ROOT.TH1D(
            f"fGenCt_{split}_" + key, ";#it{c}t (cm);Entries", len(ct_bins) - 1, ct_bins)
        h_rec_radius[f"{split}_" + key] = ROOT.TH1D(
            f"fRecRadius_{split}_" + key, ";#it{R}_{#it{abs}} (cm);Entries", 1000, 0, 1000)
        h_rec_ct[f"{split}_" + key] = ROOT.TH1D(
            f"fRecCt_{split}_" + key, ";#it{c}t (cm);Entries", len(ct_bins) - 1, ct_bins)


# read tree
data_frame_he3 = ROOT.RDataFrame('STree', mc_file)
data_frame_he3 = data_frame_he3.Range(0, int(1e7))
data_frame_he3 = data_frame_he3.Filter(
    'pt > 2. and pt < 10. and (flag & 1)==1')
np_he3 = data_frame_he3.AsNumpy(["pt", "pdg", "absCt", "eta"])

# analysis in centrality classes
counter = 0
print_step_index = 0
num_entries = len(np_he3["pt"])
print_steps = num_entries*np.arange(0, 1, 0.01)


for he3 in zip(np_he3['pt'], np_he3['pdg'], np_he3['absCt'], np_he3['eta']):

    # if counter > 10000:
    #     break
    if np.floor(counter/num_entries*100) < 99:
        if counter > print_steps[print_step_index]:
            print("Loading.... : ", np.floor(counter/num_entries*100), " %")
            print_step_index += 1

    split = "antimatter"
    if he3[1] == 1000020030:
        split = "matter"
    absCt = he3[2]
    key_counter = 0

    for key in func.keys():

        if key_counter == 0:
            key_counter += 1
        # rejection sampling to reweight pt
        if ROOT.gRandom.Rndm()*func_max[key] > func[key].Eval(he3[0]):
            continue
        # sample decay ct and ckeck for absorption
        decCt = ROOT.gRandom.Exp(7.6)
        # polar angle from eta
        tmp = abs(he3[3])
        tmp = ROOT.TMath.Exp(tmp)
        theta = 2*ROOT.TMath.ATan(tmp)  # eta = -log[tan(theta/2)]
        # momentum from transverse momentum and angle
        mom = he3[0]/ROOT.TMath.Sin(theta)
        # absorption radius
        abs_radius = absCt*mom/HE_3_MASS
        # decay radius
        dec_radius = decCt*mom/HE_3_MASS
        h_abs_ct[f"{split}_" + key].Fill(absCt)
        h_abs_radius[f"{split}_" + key].Fill(abs_radius)
        h_gen_radius[f"{split}_" + key].Fill(dec_radius)
        h_gen_ct[f"{split}_" + key].Fill(decCt)
        # print('gen: ', he3[0])
        if(decCt < absCt or absCt < 0.5):  # decCt < absCt
            h_rec_radius[f"{split}_" + key].Fill(dec_radius)
            h_rec_ct[f"{split}_" + key].Fill(decCt)
            h_rec_ct["all_" + key].Fill(decCt)

    counter += 1

for key in h_rec_ct.keys():
    key_cent = re.findall(r"[-+]?\d*\.\d+|\d+", key)
    if not outfile.GetDirectory(f"{key_cent[0]}_{key_cent[1]}"):
        outfile.mkdir(f"{key_cent[0]}_{key_cent[1]}")
    outfile.cd(f"{key_cent[0]}_{key_cent[1]}")

    # eff radius
    h_rec_radius[key].Divide(h_gen_radius[key])
    h_rec_radius[key].GetXaxis().SetTitle("#it{R}_{#it{abs}} (cm)")
    h_rec_radius[key].GetYaxis().SetTitle("1 - #it{f}_{abs}")
    h_rec_radius[key].Write("fEffRadius_" + key)
    # eff ct
    h_rec_ct[key].Divide(h_gen_ct[key])
    h_rec_ct[key].GetXaxis().SetTitle("#it{c}t (cm)")
    h_rec_ct[key].GetYaxis().SetTitle("1 - #it{f}_{abs}")
    h_rec_ct[key].Write("fEffCt_" + key)

    h_abs_ct[key].Write("fAbsCt_" + key)
    h_abs_radius[key].Write("fAbsRadius_" + key)


h_rec_ct["matter_cent_0_90_func_BGBW"].Divide(h_rec_ct["antimatter_cent_0_90_func_BGBW"])
h_rec_ct["matter_cent_0_90_func_BGBW"].Write('matter_antimatter_ratio')

outfile.Close()
