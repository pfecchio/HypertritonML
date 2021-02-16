#!/usr/bin/env python3
import argparse
import os
import time

import hyp_analysis_utils as hau
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import ROOT
import yaml
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from scipy import stats

ROOT.gROOT.SetBatch()
ROOT.ROOT.EnableImplicitMT()
ROOT.RooMsgService.instance().setSilentMode(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

# set generator seed
np.random.seed(42)

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
def chi2_to_pvalue(chi2, ndf):
    return stats.chi2.pdf(chi2, ndf)

def fisher(p_array):
    return -2*np.sum(np.log(p_array))

def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, xlimits):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size':30}
    )
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    x_obs = np.array([[res['params']['bandwidth']] for res in optimizer.res])
    y_obs = np.array([res['target'] for res in optimizer.res])
    
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
    axis.set_xlim(xlimits)
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('bandwidth', fontdict={'size':20})
    
    utility_function = UtilityFunction(kind='ucb', kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim(xlimits)
    # acq.set_ylim((np.min(utility) - 0.5, np.max(utility) + 0.5))
    acq.set_ylim((None, None))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('bandwidth', fontdict={'size':20})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    fig.savefig('gp.png', dpi=300)

###############################################################################
# define analysis global variables
CENT_CLASSES = params['CENTRALITY_CLASS']
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']

# SPLIT_MODE = args.split
SPLIT_MODE = False

SPLIT_CUTS = ['']
SPLIT_LIST = ['']
if SPLIT_MODE:
    SPLIT_LIST = ['_matter','_antimatter']
    SPLIT_CUTS = ['&& ArmenterosAlpha > 0', '&& ArmenterosAlpha < 0']

KDE_SAMPLE_SIZE = 20000
N_BINS = 70

BANDWIDTH_SPACE = (0.01, 4.)
###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
###############################################################################

standard_selection = f'm < 3.01 && m > 2.975 && V0CosPA > 0.99995 && NpidClustersHe3 > 100 && He3ProngPt > 1.8 && pt > 2 && pt < 10 && PiProngPt > 0.15 && He3ProngPvDCA > 0.05 && PiProngPvDCA > 0.2 && TPCnSigmaHe3 < 3. && TPCnSigmaHe3 > -3. && ProngsDCA < 1 && centrality >= {CENT_CLASSES[0][0]} && centrality < {CENT_CLASSES[0][1]} && ct<{CT_BINS[-1]} && ct>{CT_BINS[0]}'

###############################################################################
start_time = time.time()
###############################################################################

###############################################################################
rdf_mc = ROOT.RDataFrame('SignalTable', signal_path)

mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2.975, 3.01, 'GeV/c^{2}')
###############################################################################

for split, splitcut in zip(SPLIT_LIST, SPLIT_CUTS):
    df_mc = rdf_mc.Filter(standard_selection + splitcut)

    # define global RooFit objects
    for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
        mass_array_mc = df_mc.Filter(f'pt<{ptbin[1]} && pt>{ptbin[0]}').AsNumpy(['m'])['m']
        np.random.shuffle(mass_array_mc)

        slice_kde, slice_valid = np.array_split(mass_array_mc, 2)

        roo_slice_valid = hau.ndarray2roo(slice_valid, mass)

        nchunks = int(len(slice_kde) / KDE_SAMPLE_SIZE)
        chunks_kde = np.array_split(slice_kde, nchunks)

        def cv_score(bandwidth):
            chi2_array = []
            # for chunk_kde, chunk_valid in zip(chunks_kde, chunks_valid):
            for chunk_kde in chunks_kde:
                roo_chunk_kde = hau.ndarray2roo(chunk_kde, mass)
                # roo_chunk_valid = hau.ndarray2roo(chunk_valid, mass)

                signal = ROOT.RooKeysPdf('signal', 'signal', mass, roo_chunk_kde, ROOT.RooKeysPdf.NoMirror, bandwidth)

                frame = mass.frame(N_BINS)
                frame.SetName(f'pt{ptbin[0]}{ptbin[1]}_bandwidth{bandwidth}')

                roo_slice_valid.plotOn(frame, ROOT.RooFit.Name('test_data'))
                signal.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Name('signal'))

                chi2 = frame.chiSquare('signal', 'test_data') * N_BINS
                frame.Write()

                chi2_array.append(chi2_to_pvalue(chi2, N_BINS))

            score = -fisher(chi2_array) / (2 * len(chi2_array))
            if score < -100:
                score = -100

            return score

        bandwidth_space = {'bandwidth': BANDWIDTH_SPACE}
                
        optimizer = BayesianOptimization(f=cv_score, pbounds=bandwidth_space, random_state=42)
        optimizer.maximize(init_points=5, n_iter=10)

        # plot_gp(optimizer, np.linspace(BANDWIDTH_SPACE[0], BANDWIDTH_SPACE[1], 10000).reshape(-1, 1), BANDWIDTH_SPACE)

###############################################################################
print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')

