import pandas as pd
import uproot 
import numpy as numpy
import matplotlib.pyplot as plt
import os
import pickle
import xgboost as xgb
import analysis_utils as au
df_bkg=uproot.open(os.environ['HYPERML_TABLES']+'/BkgTable.root')['BkgTable'].pandas.df()
df_cut=df_bkg.query('Ct<2 and Centrality<10')
filename='/BDT_Ct_0.00_2.00_pT_0.00_10.00_Cen_0.00_10.00.sav'
model= model = pickle.load(open(os.environ['HYPERML_MODELS_2']+filename, 'rb'))
training_columns = [ 'V0CosPA','ProngsDCA','PiProngPvDCAXY','He3ProngPvDCAXY','HypCandPt','ArmenterosAlpha','NpidClustersHe3','TPCnSigmaHe3','He3ProngPvDCA','PiProngPvDCA']
data = xgb.DMatrix(data=(df_cut[training_columns]))
y_pred = model.predict(data,output_margin=True)
df_cut.eval('Score = @y_pred',inplace=True)
df_cut.eval('y = 0',inplace=True)
au.plot_corr(df_cut,training_columns+['InvMass'],'Background')
plt.hist(df_cut.query('Score>3')['InvMass'],bins=20,range=[2.96,3.05])
plt.show()
