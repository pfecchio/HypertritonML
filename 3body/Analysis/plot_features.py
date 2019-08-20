import os

import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.index import Index

import analysis_utils as au
import generalized_analysis as ga


training_columns = [
    'HypCandPt', 'PtDeu', 'PtP', 'PtPi', 'nClsTPCDeu', 'nClsTPCP', 'nClsTPCPi', 'nClsITSDeu', 'nClsITSP',
    'nClsITSPi', 'nSigmaTPCDeu', 'nSigmaTPCP', 'nSigmaTPCPi', 'nSigmaTOFDeu', 'nSigmaTOFP', 'nSigmaTOFPi',
    'trackChi2Deu', 'trackChi2P', 'trackChi2Pi', 'vertexChi2', 'DCA2xyPrimaryVtxDeu', 'DCAxyPrimaryVtxP',
    'DCAxyPrimaryVtxPi', 'DCAzPrimaryVtxDeu', 'DCAzPrimaryVtxP', 'DCAzPrimaryVtxPi', 'DCAPrimaryVtxDeu',
    'DCAPrimaryVtxP', 'DCAPrimaryVtxPi', 'DCAxyDecayVtxDeu', 'DCAxyDecayVtxP', 'DCAxyDecayVtxPi', 'DCAzDecayVtxDeu',
    'DCAzDecayVtxP', 'DCAzDecayVtxPi', 'DCADecayVtxDeu', 'DCADecayVtxP', 'DCADecayVtxPi', 'TrackDistDeuP',
    'TrackDistPPi', 'TrackDistDeuPi', 'CosPA']  # 42

table_path = os.environ['HYPERML_TABLES_3']
signal_table_path = '{}/HyperTritonTable_19d2.root'.format(table_path)
background_table_path = '{}/HyperTritonTable_18q.root'.format(table_path)

analysis = ga.GeneralizedAnalysis(3, signal_table_path, background_table_path)

# get the data for the features visualization
bkg = analysis.df_data
sig = analysis.df_signal

df = pd.concat([bkg, sig])

# too many features, divide into chunks
features_chunks = []

ind_a = 0
ind_b = len(training_columns)
for i in range(ind_a, ind_b, 9):
    x = i
    features_chunks.append(training_columns[x:x + 9])
index = 0
for chunks in features_chunks:
    au.plot_distr(df, chunks, out_name='features{}.pdf'.format(index))
    index = index + 1
    
