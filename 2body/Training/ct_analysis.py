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
from ROOT import TH2D,TH1D,TCanvas,TFile,TF1,gStyle,TPaveText
from array import array
import analysis_utils as au

file_name = os.environ['HYPERML_FIGURES']+'/results.txt'
Training = True
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
      model =Analysis.TrainingAndTest(training_columns,params_def,Ct_bins[index_ct],[0,10],Centrality_bins[index_cen],draw=False,optimize=False)
      if model is not 0:
        filename = '/BDT_Ct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_Cen_{:.2f}_{:.2f}.sav'.format(Ct_bins[index_ct][0],Ct_bins[index_ct][1],0,10,Centrality_bins[index_cen][0],Centrality_bins[index_cen][1])
        pickle.dump(model, open(os.environ['HYPERML_MODELS']+filename, 'wb'))
        Cut,eff = Analysis.Significance(model,training_columns,Ct_bins[index_ct],[0,10],Centrality_bins[index_cen],draw=False,custom=False,score_shift=0.7)
        Cut_saved.append(Cut)
        Eff_BDT.append(eff)
        print(filename+' has been saved')
else:
  Cut_saved = [3.7323232323232327, 4.984848484848485, 4.984848484848485, 3.262626262626263, 4.202020202020202, 5.141414141414142, 5.924242424242426, 2.7929292929292933, 4.202020202020202, 5.297979797979799, 5.924242424242426, 3.575757575757576, 4.515151515151516, 5.454545454545455, 5.924242424242426, 2.94949494949495, 4.202020202020202, 5.611111111111111, 5.297979797979799, 2.6363636363636367, 5.141414141414142, 6.237373737373737, 6.3939393939393945, 4.671717171717172, 4.828282828282829, 6.3939393939393945, 6.080808080808081, 5.297979797979799, 5.297979797979799, 6.550505050505052, 6.3939393939393945, 4.358585858585859, 4.515151515151516, 5.767676767676768, 5.924242424242426, 5.454545454545455]
  Eff_BDT = [0.5915899850203296, 0.7018650328858456, 0.7943463544049035, 0.8878878878878879, 0.7549668874172185, 0.8645517168571353, 0.8526714513556619, 0.9465791940018744, 0.8005347897821016, 0.8875702479338843, 0.8800990749633524, 0.903954802259887, 0.7922886452069331, 0.8908267650511488, 0.8891578878492442, 0.9411764705882353, 0.8354150057578174, 0.8742990159771453, 0.9224039247751431, 0.9495495495495495, 0.735659111054831, 0.8116084977238239, 0.8276492801510503, 0.8386243386243386, 0.7296039706338537, 0.7785635898061212, 0.8258119335347432, 0.6937901498929336, 0.6167242570836213, 0.702062120304612, 0.7194157845370495, 0.7146739130434783, 0.6180788050476436, 0.6996820349761527, 0.7374109555391796, 0.4368421052631579]



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

results.cd()
cv = TCanvas("ct")

histoct = TH1D("histoct",";ct [cm];dN/dct [cm^{-1}]",len(ct_binning)-1,ct_binning)


gStyle.SetOptStat(0)
gStyle.SetOptFit(0)
  ####################

histoct.UseCurrentStyle()
histoct.SetLineColor(1)
histoct.SetMarkerStyle(20)
histoct.SetMarkerColor(1)

for index in range(0,len(Fit_counts)):
  histoct.SetBinContent(index+1,Fit_counts[index][0]/Effp[index]/(ct_binning[index+1]-ct_binning[index]))
  histoct.SetBinError(index+1,Fit_counts[index][1]/Effp[index]/(ct_binning[index+1]-ct_binning[index]))


expo = TF1("","[0]*exp(-x/[1]/0.029979245800)")
expo.SetParLimits(1,180,240)
histoct.Fit(expo,"M")

pinfo2= TPaveText(0.5,0.5,0.91,0.9,"NDC")
pinfo2.SetBorderSize(0)
pinfo2.SetFillStyle(0)
pinfo2.SetTextAlign(30+3)
pinfo2.SetTextFont(42)
string ='ALICE Internal, Pb-Pb 2018 {}-{}'.format(0,90)
pinfo2.AddText(string)
string='#tau = {:.0f} #pm {:.0f} ps '.format(expo.GetParameter(1),expo.GetParError(1))
pinfo2.AddText(string)  
pinfo2.Draw()
  
histoct.Write()
cv.Write()
results.Close()
