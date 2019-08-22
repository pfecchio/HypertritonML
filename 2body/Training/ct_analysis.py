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
from ROOT import TH2D,TH1D,TCanvas,TFile,TF1
from array import array
import analysis_utils as au

file_name = os.environ['HYPERML_FIGURES']+'/results.txt'
Training = False
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

if Training is True:
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
else:
  Cut_saved = [3.8888888888888893, 5.454545454545455, 5.141414141414142, 3.4191919191919196, 4.045454545454546, 5.454545454545455, 5.611111111111111, 3.7323232323232327, 4.202020202020202, 5.454545454545455, 5.924242424242426, 2.94949494949495, 4.515151515151516, 5.924242424242426, 6.080808080808081, 3.1060606060606064, 4.358585858585859, 5.611111111111111, 5.611111111111111, 4.202020202020202, 4.671717171717172, 6.863636363636363, 6.3939393939393945, 4.984848484848485, 4.515151515151516, 6.080808080808081, 6.863636363636363, 5.141414141414142, 5.141414141414142, 6.550505050505052, 7.020202020202021, 5.767676767676768, 4.828282828282829, 6.863636363636363, 6.550505050505052, 4.358585858585859] 
  Eff_BDT = [0.6229304892416444, 0.7309497425200443, 0.862548301299744, 0.9064748201438849, 0.7939792008757526, 0.8808792722472523, 0.914352581124263, 0.9306029579067122, 0.837192233268833, 0.908523531332777, 0.9223050458715596, 0.9521739130434783, 0.8198940553266627, 0.894154449528434, 0.9090508698524554, 0.9464944649446494, 0.8480246389124894, 0.9124711316397228, 0.9333149374540103, 0.92, 0.8131095486504877, 0.8188827892553325, 0.8762599469496021, 0.8595600676818951, 0.8091885883660616, 0.8633420121503438, 0.8221696591764326, 0.8210526315789474, 0.7168874172185431, 0.7997373604727511, 0.7555086024750981, 0.6342281879194631, 0.6372764355192971, 0.6108507570295602, 0.7518902268272193, 0.7571428571428571]


print("efficiency BDT: ",Eff_BDT)
print("cut: ",Cut_saved)


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

print('counts : ',Ct_counts)


#Ct_counts = [108.8193359375, 141.8958282470703, 107.75634765625, 96.154296875, 68.265625, 50.67390441894531, 22.83233642578125, 21.54931640625, 7.135498046875]


#loop to compute the efficiency
Effp = []
for index in range(0,len(Ct_bins)):
  Effp.append(Analysis.EfficiencyPresel(Ct_bins[index],pt_cut=[0,10],centrality_cut=[0,100]))
  Ct_counts[index]=Ct_counts[index]/Effp[index]
errCt = np.sqrt(Ct_counts)

ct_binning = array("d",[0,2,4,6,8,10,14,18,23,28])
results = TFile(os.environ['HYPERML_DATA']+"/results.root","update")
histoct = TH1D("histoct",";ct [cm];dN/dct [cm^{-1}]",len(ct_binning)-1,ct_binning)
for index in range(0,len(Fit_counts)):
  histoct.SetBinContent(index,Fit_counts[index][0]/Effp[index]/(ct_binning[index+1]-ct_binning[index]))
  histoct.SetBinError(index,Fit_counts[index][1]/Effp[index]/(ct_binning[index+1]-ct_binning[index]))

expo = TF1("","[0]*exp(-x/[1]/0.029979245800)")
expo.SetParLimits(1,180,240)
histoct.Fit(expo,"M")
histoct.Write()