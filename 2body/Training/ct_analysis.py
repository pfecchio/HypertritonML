import training_utils as tu
import os
import pickle
import xgboost as xgb
import uproot
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

params_def = {
    # Parameters that we are going to tune.
    'max_depth':8,
    'eta':0.05,
    'gamma':0.7,
    'min_child_weight':8,
    'subsample':0.8,
    'colsample_bytree':0.9,
    'objective':'binary:logistic',
    'random_state':42,
    'silent':1,
    'nthread':4,
    'tree_method':'hist',
    'scale_pos_weight': 10}

training_columns = [ 'V0CosPA','ProngsDCA', 'DistOverP','ArmenterosAlpha','NpidClustersHe3','V0pt','TPCnSigmaHe3','He3ProngPvDCA','PiProngPvDCA']

Analysis = tu.Generalized_Analysis('../DerivedTrees/SignalTable.root','../DerivedTrees/DataTable.root','ProngsDCA<1.6 and He3ProngPvDCA>0.01 and He3ProngPvDCA>0.01 and V0CosPA>0.98','(InvMass<2.98 or InvMass>3.005) and V0pt<=10')
 

loop over all the bins
if not os.path.exists('Models'):
  os.makedirs('Models')
Centrality_bins = [[0,10],[10,30],[30,50],[50,90]]
Pt_bins = [[2,3],[3,4],[4,5],[5,9]]
Ct_bins = [[0,2],[2,4],[4,6],[6,8],[8,10],[10,14],[14,18],[18,23],[23,28]]

## loop to train the models
# if not os.path.exists('Models'):
#   os.makedirs('Models')
# Centrality_bins = [[0,10],[10,30],[30,50],[50,90]]
# Ct_bins = [[0,2],[2,4],[4,6],[6,8],[8,10],[10,14],[14,18],[18,23],[23,28]]
# for index_cen in range(0,len(Centrality_bins)):
#     for index_ct in range(0,len(Ct_bins)):
#         print('centrality: ',Centrality_bins[index_cen])
#         print('Ct: ',Ct_bins[index_ct])
#         model =Analysis.TrainingAndTest(training_columns,params_def,Ct_bins[index_ct],[0,10],Centrality_bins[index_cen],draw=False)
#         if model is not 0:
#             filename = 'Models/BDT_Ct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_Cen_{:.2f}_{:.2f}.sav'.format(Ct_bins[index_ct][0],Ct_bins[index_ct][1],0,10,Centrality_bins[index_cen][0],Centrality_bins[index_cen][1])
#             pickle.dump(model, open(filename, 'wb'))
#             print(filename+' has been saved')


## loop to read the models and to do the prediction

# for index_ct in range(0,len(Ct_bins)):
#   for index_cen in range(0,len(Centrality_bins)):
#     print('centrality: ',Centrality_bins[index_cen])
#     print('Ct: ',Ct_bins[index_ct])
#     #model =Analysis.TrainingAndTest(training_columns,params_def,Ct_bins[index_ct],[0,10],Centrality_bins[index_cen],draw=False)
    
#     ct_min = Ct_bins[index_ct][0]
#     ct_max = Ct_bins[index_ct][1]
#     centrality_min = Centrality_bins[index_cen][0]
#     centrality_max = Centrality_bins[index_cen][1]

#     total_cut = '@ct_min<Ct<@ct_max and 0<V0pt<10 and @centrality_min<Centrality<@centrality_max'

#     filename = 'Models/BDT_Ct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_Cen_{:.2f}_{:.2f}.sav'.format(Ct_bins[index_ct][0],Ct_bins[index_ct][1],0,10,Centrality_bins[index_cen][0],Centrality_bins[index_cen][1])
#     model = pickle.load(open(filename, 'rb'))
#     dfDataF = Analysis.dfData.query(total_cut)
#     data = xgb.DMatrix(data=(dfDataF[training_columns]))
#     y_pred = model.predict(data,output_margin=True)
#     dfDataF.eval('Score = @y_pred',inplace=True)
#     Counts,bins = np.histogram(dfDataF.query('Score >@output_cut')['InvMass'],bins=26,range=[2.96,3.05])
#     #sum over the counts in centrality intervals
#     if index_cen==0:
#       CountsTot=Counts
#     else:
#       CountsTot=CountsTot+Counts
    
#   plt.errorbar(bins[:-1], CountsTot, yerr=1,xerr=(bins[2]-bins[1])/2, fmt='o', c='b')
#   filename = 'InvMass_Ct_{:.2f}_{:.2f}'.format(Ct_bins[index_ct][0],Ct_bins[index_ct][1],0,10,Centrality_bins[index_cen][0],Centrality_bins[index_cen][1])
#   plt.savefig('Peaks/'+filename+'.pdf')
#   plt.close()