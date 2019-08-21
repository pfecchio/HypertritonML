import training_utils as tu
import os
import pickle
import xgboost as xgb
import uproot
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import scipy
import os
from ROOT import TH2D,TH1D,TCanvas,TFile

import analysis_utils as au

file_name = os.environ['HYPERML_FIGURES']+'/results.txt'

def gauss(x,n,mu,sigma):
    return n*np.exp(-(x-mu)**2/(2*sigma**2))
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

training_columns = [ 'V0CosPA','ProngsDCA','V0pt','ArmenterosAlpha','NpidClustersHe3','TPCnSigmaHe3','He3ProngPvDCA','PiProngPvDCA']

Analysis = tu.Generalized_Analysis(os.environ['HYPERML_TABLES']+'/SignalTable.root',os.environ['HYPERML_TABLES']+'/DataTable.root','ProngsDCA<1.6 and PiProngPvDCA>0.01 and He3ProngPvDCA>0.01 and V0CosPA>0.98 and 0.4<V0radius<200 and TPCnSigmaHe3<5 and NpidClustersHe3>=50 and NpidClustersPion>=50 and He3ProngPt>1.2','(InvMass<2.98 or InvMass>3.005) and V0pt<=10')

# loop to train the models
if not os.path.exists(os.environ['HYPERML_MODELS']):
  os.makedirs(os.environ['HYPERML_MODELS'])
Centrality_bins = [[0,10],[10,30],[30,50],[50,90]]
Ct_bins = [[0,2],[2,4],[4,6],[6,8],[8,10],[10,14],[14,18],[18,23],[23,28]]

Cut_saved = []
Eff_BDT = []
for index_ct in range(0,len(Ct_bins)):
  for index_cen in range(0,len(Centrality_bins)):
    print('centrality: ',Centrality_bins[index_cen])
    print('Ct: ',Ct_bins[index_ct])
    model =Analysis.TrainingAndTest(training_columns,params_def,Ct_bins[index_ct],[2,10],Centrality_bins[index_cen],draw=False,optimize=False)
    if model is not 0:
      filename = '/BDT_Ct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_Cen_{:.2f}_{:.2f}.sav'.format(Ct_bins[index_ct][0],Ct_bins[index_ct][1],0,10,Centrality_bins[index_cen][0],Centrality_bins[index_cen][1])
      pickle.dump(model, open(os.environ['HYPERML_MODELS']+filename, 'wb'))
      Cut,eff = Analysis.Significance(model,training_columns,Ct_bins[index_ct],[2,10],Centrality_bins[index_cen],draw=False)
      Cut_saved.append(Cut)
      Eff_BDT.append(eff)
      print(filename+' has been saved')

# print("efficiency BDT: ",Eff_BDT)
# print("cut: ",Cut_saved)


Ct_counts = []
Fit_counts = []
# loop to read the models and to do the prediction
index_cut = 0
plt.close()
if not os.path.exists(os.environ['HYPERML_FIGURES']+'/Peaks/'):
  os.makedirs(os.environ['HYPERML_FIGURES']+'/Peaks/')
for index_ct in range(0,len(Ct_bins)):
  for index_cen in range(0,len(Centrality_bins)):
    output_cut = Cut_saved[index_cut]
    print('centrality: ',Centrality_bins[index_cen],'Ct: ',Ct_bins[index_ct])
    
    ct_min = Ct_bins[index_ct][0]
    ct_max = Ct_bins[index_ct][1]
    centrality_min = Centrality_bins[index_cen][0]
    centrality_max = Centrality_bins[index_cen][1]

    total_cut = '@ct_min<Ct<@ct_max and 0<V0pt<10 and @centrality_min<Centrality<@centrality_max'
    filename = '/BDT_Ct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_Cen_{:.2f}_{:.2f}.sav'.format(Ct_bins[index_ct][0],Ct_bins[index_ct][1],0,10,Centrality_bins[index_cen][0],Centrality_bins[index_cen][1])
    model = pickle.load(open(os.environ['HYPERML_MODELS']+filename, 'rb'))
    dfDataF = Analysis.dfData.query(total_cut)
    data = xgb.DMatrix(data=(dfDataF[training_columns]))
    y_pred = model.predict(data,output_margin=True)
    dfDataF.eval('Score = @y_pred',inplace=True)
    Counts,bins = np.histogram(dfDataF.query('Score >@output_cut')['InvMass'],bins=26,range=[2.96,3.05])
    #sum over the counts in centrality intervals
    if index_cen==0:
      CountsTot=Counts/Eff_BDT[index_ct*len(Centrality_bins)+index_cen]
    else:
      CountsTot=CountsTot+Counts/Eff_BDT[index_ct*len(Centrality_bins)+index_cen]
    index_cut=index_cut+1
  
  
  # to save the plots
  recreate=False
  if index_ct is 0:
    recreate=True
  Fit_counts.append(au.fit(CountsTot,Ct_bins[index_ct][0],Ct_bins[index_ct][1],recreate=recreate))
  
  ##
  bin_centers = 0.5*(bins[1:]+bins[:-1])
  sidemap = (bin_centers<2.975) + (bin_centers>3.005)
  massmap = np.logical_not(sidemap)
  bins_side = bin_centers[sidemap]
  counts_side = CountsTot[sidemap]
  h, residuals, _, _, _ = np.polyfit(bins_side,counts_side,2,full=True)
  y = np.polyval(h,bins_side)
  bkg = sum(np.polyval(h,bin_centers[massmap]))
  sig = sum(CountsTot[massmap])-bkg

  Ct_counts.append(sig/(Ct_bins[index_ct][1]-Ct_bins[index_ct][0]))

  plt.errorbar(bins[:-1], CountsTot, yerr=np.sqrt(CountsTot),xerr=(bins[2]-bins[1])/2, fmt='o', c='b')
  plt.plot(bins[:-1],np.polyval(h,bins[:-1]),'g--')
  plt.axvline(x=2.975)
  plt.axvline(x=3.005)

  filename = 'InvMass_Ct_{:.2f}_{:.2f}.pdf'.format(Ct_bins[index_ct][0],Ct_bins[index_ct][1])
  plt.savefig(os.environ['HYPERML_FIGURES']+'/Peaks/'+filename)
  plt.close()
histo.Write()
results.Close()
print('counts : ',Ct_counts)


#Ct_counts = [108.8193359375, 141.8958282470703, 107.75634765625, 96.154296875, 68.265625, 50.67390441894531, 22.83233642578125, 21.54931640625, 7.135498046875]
bins = [1,3,5,7,9,12,16,20.5,25.5]
errbins = [1,1,1,1,1,2,2,2.5,2.5]
#loop to compute the efficiency
Effp = []
for index in range(0,len(Ct_bins)):
  Effp.append(Analysis.EfficiencyPresel(Ct_bins[index],pt_cut=[0,10],centrality_cut=[0,100]))
  Ct_counts[index]=Ct_counts[index]/Effp[index]
errCt = np.sqrt(Ct_counts)
print('eff presel: ',Effp)
def expo(x,n,tau):
    return n*np.exp(-x/tau/0.029979245800)

print('counts corrected: ',Ct_counts)
fig, ax = plt.subplots()
plt.errorbar(bins, Ct_counts, yerr=errCt,xerr=errbins, fmt='o', c='b')
par,cov = curve_fit(expo,bins,Ct_counts,bounds=([1800,180],[8000,240]))
plt.plot(bins,expo(np.array(bins),*par),'r--',label='fit: N(0)={:.2f}, $\\tau$={:.2f}$\pm${:.2f}ps'.format(par[0],par[1],cov[1][1]))
ax.set_yscale('log')
plt.xlabel('ct [cm]')
plt.ylabel('$\\frac{dN}{dct}$ [cm$^-1$]')
print('N : ',par[0])
print('tau : ',par[1],' +- ',math.sqrt(cov[1][1]), ' ps')
plt.savefig(os.environ['HYPERML_FIGURES']+'/Peaks/'+'ct.pdf')
plt.show()