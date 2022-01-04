import yaml
import argparse
import os
import numpy as np
import ROOT


###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


###############################################################################

# define some globals
results_dir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]
FILE_PREFIX = params['FILE_PREFIX']

os.system('python3 compute_lifetime.py ../Config/2body_analysis_new.yaml -s -syst --matter')
os.system('python3 compute_lifetime.py ../Config/2body_analysis_new.yaml -s -syst --antimatter')

arr_matter = np.load(results_dir + f'/{FILE_PREFIX}_matter_tau_syst_array.npy')
arr_antimatter = np.load(results_dir + f'/{FILE_PREFIX}_antimatter_tau_syst_array.npy')

arr_matter = arr_matter[arr_matter>0]
arr_antimatter = arr_antimatter[arr_antimatter>0]

h_asymmetry_distribution = ROOT.TH1D(f'fAsymmetryDistribution', '', 400, -200, 200)

N_trials = 10000
n=0

while n<N_trials:
    tau_mat = np.random.choice(arr_matter)
    tau_antimat = np.random.choice(arr_antimatter)
    h_asymmetry_distribution.Fill(tau_mat - tau_antimat)
    n+=1

h_asymmetry_distribution.GetXaxis().SetTitle("Asymmetry (ps)")
h_asymmetry_distribution.GetYaxis().SetTitle("Entries")
file = ROOT.TFile('asym.root', 'recreate')
h_asymmetry_distribution.Write()
file.Close()
