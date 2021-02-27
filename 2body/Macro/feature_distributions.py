#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import seaborn as sns
import uproot
from hipe4ml import plot_utils

df_sig = uproot.open(os.environ['HYPERML_TABLES_2'] + '/SignalTable_20g7.root')['SignalTable'].pandas.df()
df_bkg = uproot.open(os.environ['HYPERML_TABLES_2'] + '/DataTable_18LS.root')['DataTable'].pandas.df()

training_columns = ['ProngsDCA','He3ProngPvDCA','He3ProngPvDCAXY','PiProngPvDCA','PiProngPvDCAXY','TPCnSigmaHe3','TPCnSigmaPi','NpidClustersHe3','NitsClustersHe3','V0CosPA', 'pt']

training_labels = [r'$\mathrm{DCA_{daughters}}$ (cm)', r'$\mathrm{DCA_{PV} \/ ^{3}He} $ (cm)',  r'$\mathrm{DCA_{PV} \/ \pi} $ (cm)', r'$\mathrm{DCA_{PV XY} \/ ^{3}He}$ (cm)',  r'$\mathrm{DCA_{PV XY} \/ \pi}$ (cm)', r'n$\sigma_{\mathrm{TPC}} \/ \mathrm{^{3}He}$',  r'n$\sigma_{\mathrm{TPC}} \/ \mathrm{\pi}$', r'n$_{\mathrm{cluster TPC}} \/ \mathrm{^{3}He}$', r'n$_{\mathrm{cluster ITS}} \/ \mathrm{^{3}He}$',r'cos($\theta_{\mathrm{pointing}}$)', r'$p_\mathrm{T}$ (GeV/$c$)']  

bins= [80, 63, 63, 63, 63, 79, 78, 127, 63, 63, 63]
log_scale = [True, True, True, True, True, True, True, True, False, True, True]

fig, axs = plt.subplots(3,4, figsize=(35, 22))
axs = axs.flatten()

for index, variable in enumerate(training_columns, start=0):
    ax = axs[index]
    ax.grid(True)
    ax = sns.distplot(df_sig[variable], norm_hist=True, kde=False, bins=bins[index], hist_kws={'log': log_scale[index]}, label='Signal', ax=ax)
    ax = sns.distplot(df_bkg[variable], norm_hist=True, kde=False, bins=bins[index], hist_kws={'log': log_scale[index]}, label='Background', ax=ax)
    ax.set_xlabel(training_labels[index], fontsize=30)
    ax.set_ylabel('counts (arb. units)', fontsize=30)
    ax.set_xlim(df_bkg[variable].min(), df_sig[variable].max())
    ax.tick_params(direction='in')

fig.delaxes(axs[-1])
fig.delaxes(axs[-2])
# axs[-4].legend(bbox_to_anchor=(3.9, 0.58),prop={'size': 48}, frameon=False)
# plt.text(0.61, 0.31, 'ALICE Performance', fontsize=48, transform=plt.gcf().transFigure)
# plt.text(0.595, 0.263, 'Pb-Pb $\sqrt{s_{\mathrm{NN}}} = $ 5.02TeV ', fontsize=48, transform=plt.gcf().transFigure)

fig.savefig(os.environ['HYPERML_FIGURES_2'] + '/feature_distribution_2body.png',bbox_inches='tight')
# fig.savefig(os.environ['HYPERML_FIGURES_2'] + '/feature_distribution_2body.pdf',bbox_inches='tight')
# fig.set_rasterized(True)
# fig.savefig(os.environ['HYPERML_FIGURES_2'] + '/feature_distribution_2body.eps', format='eps', bbox_inches='tight')

