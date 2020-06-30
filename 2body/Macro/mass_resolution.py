import uproot
import pandas as pd
from ROOT import TF1, TH1, TH1D, TH2D, TFile, gDirectory
import matplotlib.pyplot as plt
import numpy as np
import math
import yaml
import argparse
import os

hyp3mass = 2.99131

parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to the YAML configuration file")
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def h1_mass_res(counts, cent_class, pt_range, ct_range, bins=45, name=''):
    th1 = TH1D(f'ct{ct_range[0]}{ct_range[1]}_pT{pt_range[0]}{pt_range[1]}_cen{cent_class[0]}{cent_class[1]}_{name}', '', 100, -0.03, 0.03)

    for index in range(0, len(counts)):
        th1.SetBinContent(index+1, counts[index])
        th1.SetBinError(index + 1, math.sqrt(counts[index]))

    th1.SetDirectory(0)

    return th1

df = uproot.open(os.path.expandvars(params['MC_PATH']))["SignalTable;4"].pandas.df()
df['mass_shift'] = df.apply(lambda row: row.m - hyp3mass, axis = 1)

ct_bins = params['CT_BINS']

resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]
file_name =  resultsSysDir + '/' + params['FILE_PREFIX'] + '_mass_res.root'
results_file = TFile(file_name,"recreate")

hist_mean = TH1D('hist_mean', ';ct bin; #mu (GeV/c^{{2}})',len(ct_bins)-1,0,len(ct_bins))
fit_mean = TH1D('fit_mean', ';ct bin; #mu (GeV/c^{{2}})',len(ct_bins)-1,0,len(ct_bins))
gauss = TF1("gauss","gaus",-0.002,0.002)

for bin in range(1,len(ct_bins)):
    dfq = df.query('ct>@ct_bins[@bin-1] and ct<@ct_bins[@bin]')
    counts, _ = np.histogram(dfq['mass_shift'], bins=100, range=[-0.03, 0.03])
    histo = h1_mass_res(counts,[0,90],[0,10],[ct_bins[bin-1],ct_bins[bin]])
    histo.Fit(gauss,"QR")
    fit_mean.SetBinContent(bin, gauss.GetParameter(1))
    fit_mean.SetBinError(bin, gauss.GetParError(1))
    hist_mean.SetBinContent(bin, histo.GetMean())
    hist_mean.SetBinError(bin, histo.GetMeanError())
    histo.Write()

fit_mean.Write()
hist_mean.Write()