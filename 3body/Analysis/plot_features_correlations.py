import os

import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.index import Index

import plot_utils as pu
import generalized_analysis as ga


TRAINING_COLUMNS = [
    'PtDeu', 'PtP', 'PtPi', 'nClsTPCDeu', 'nClsTPCP', 'nClsTPCPi', 'nClsITSDeu', 'nClsITSP', 'nClsITSPi',
    'nSigmaTPCDeu', 'nSigmaTPCP', 'nSigmaTPCPi', 'nSigmaTOFDeu', 'nSigmaTOFP', 'nSigmaTOFPi', 'DCA2xyPrimaryVtxDeu',
    'DCAxyPrimaryVtxP', 'DCAxyPrimaryVtxPi', 'DCAzPrimaryVtxDeu', 'DCAzPrimaryVtxP', 'DCAzPrimaryVtxPi',
    'DCAPrimaryVtxDeu', 'DCAPrimaryVtxP', 'DCAPrimaryVtxPi', 'DCAxyDecayVtxDeu', 'DCAxyDecayVtxP', 'DCAxyDecayVtxPi',
    'DCAzDecayVtxDeu', 'DCAzDecayVtxP', 'DCAzDecayVtxPi', 'DCADecayVtxDeu', 'DCADecayVtxP', 'DCADecayVtxPi',
    'TrackDistDeuP', 'TrackDistPPi', 'TrackDistDeuPi', 'CosPA']  # 38

table_path = os.environ['HYPERML_TABLES_3']
signal_table_path = '{}/HyperTritonTable_19d2.root'.format(table_path)
background_table_path = '{}/HyperTritonTable_18qr.root'.format(table_path)

analysis = ga.GeneralizedAnalysis(3, signal_table_path, background_table_path)

# get the data for the features visualization
bkg = analysis.df_data
sig = analysis.df_signal

df = pd.concat([bkg, sig])

# too many features, divide into chunks
features_chunks = []

ind_a = 0
ind_b = len(TRAINING_COLUMNS)

for i in range(ind_a, ind_b, 9):
    x = i
    features_chunks.append(TRAINING_COLUMNS[x:x + 9])

index = 0
for chunks in features_chunks:
    pu.plot_distr(df, chunks, fig_name='features{}.pdf'.format(index))
    index = index + 1

TRAINING_COLUMNS.append('InvMass')

# plot correlations
pu.plot_corr(df,TRAINING_COLUMNS)
