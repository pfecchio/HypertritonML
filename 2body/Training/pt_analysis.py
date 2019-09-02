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

file_name ='resultspt.root'
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

training_columns = [ 'V0CosPA','ProngsDCA','PiProngPvDCAXY','He3ProngPvDCAXY','HypCandPt','ArmenterosAlpha','He3ProngPvDCA','PiProngPvDCA','NpidClustersHe3','TPCnSigmaHe3']

Analysis = tu.Generalized_Analysis(os.environ['HYPERML_TABLES']+'/SignalTable.root',os.environ['HYPERML_TABLES']+'/DataTable.root','2<=HypCandPt<=10','(InvMass<2.98 or InvMass>3.005) and HypCandPt<=10')

# loop to train the models
if not os.path.exists(os.environ['HYPERML_MODELS']):
  os.makedirs(os.environ['HYPERML_MODELS'])
Centrality_bins = [[0,10],[10,30],[30,50],[50,90]]
Pt_bins = [[2,4],[4,6],[6,8],[8,10]]

if Training:
  Cut_saved = []
  Eff_BDT = []
  for index_pt in range(0,len(Pt_bins)):
    for index_cen in range(0,len(Centrality_bins)):
      print('centrality: ',Centrality_bins[index_cen])
      print('pT: ',Pt_bins[index_pt])
      model =Analysis.TrainingAndTest(training_columns,params_def,[0,10],Pt_bins[index_pt],Centrality_bins[index_cen],draw=False,optimize=False)
      if model is not 0:
        filename = '/BDT_Ct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_Cen_{:.2f}_{:.2f}.sav'.format(0,100,Pt_bins[index_pt][0],Pt_bins[index_pt][1],Centrality_bins[index_cen][0],Centrality_bins[index_cen][1])
        pickle.dump(model, open(os.environ['HYPERML_MODELS']+filename, 'wb'))
        Cut,eff = Analysis.Significance(model,training_columns,[0,100],Pt_bins[index_pt],Centrality_bins[index_cen],draw=False,custom=True,score_shift=0)
        Cut_saved.append(Cut)
        Eff_BDT.append(eff)
        print(filename+' has been saved')
else:
  Eff_BDT  = [0.27472211968335997, 0.4377771845370857, 0.36194441202028454, 0.5398807834232189, 0.4695746430440308, 0.5766552058538306, 0.6111634596310256, 0.7885985748218527, 0.3449018669219722, 0.41208729714605485, 0.5105594664690626, 0.5684210526315789, 0.5594405594405595, 0.5518018018018018, 0.9273356401384083, 0.8333333333333334]
  Cut_saved = [6.0, 7.5, 8.0, 6.600000000000001, 5.6, 7.5, 7.700000000000001, 5.5, 5.9, 7.9, 7.800000000000001, 6.0, 4.6000000000000005, 5.5, 0.30000000000000027, -3.0]


print("efficiency BDT: ",Eff_BDT)
print("cut: ",Cut_saved)

Fit_counts = [[[0,0]],[[0,0]],[[0,0]],[[0,0]]]
# loop to read the models and to do the prediction
index_cut = 0
plt.close()
if not os.path.exists(os.environ['HYPERML_FIGURES']+'/Peaks/'):
  os.makedirs(os.environ['HYPERML_FIGURES']+'/Peaks/')
for index_pt in range(0,len(Pt_bins)):
  for index_cen in range(0,len(Centrality_bins)):
    output_cut = Cut_saved[index_cut]
    print('centrality: ',Centrality_bins[index_cen],'Pt: ',Pt_bins[index_pt])
    
    pt_min = Pt_bins[index_pt][0]
    pt_max = Pt_bins[index_pt][1]
    centrality_min = Centrality_bins[index_cen][0]
    centrality_max = Centrality_bins[index_cen][1]

    total_cut = '0<Ct<100 and @pt_min<HypCandPt<@pt_max and @centrality_min<Centrality<@centrality_max'
    filename = '/BDT_Ct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_Cen_{:.2f}_{:.2f}.sav'.format(0,100,Pt_bins[index_pt][0],Pt_bins[index_pt][1],Centrality_bins[index_cen][0],Centrality_bins[index_cen][1])
    model = pickle.load(open(os.environ['HYPERML_MODELS']+filename, 'rb'))
    dfDataF = Analysis.dfData.query(total_cut)
    data = xgb.DMatrix(data=(dfDataF[training_columns]))
    y_pred = model.predict(data,output_margin=True)
    dfDataF.eval('Score = @y_pred',inplace=True)
    Counts,bins = np.histogram(dfDataF.query('Score >@output_cut')['InvMass'],bins=26,range=[2.96,3.05])
    print(Counts)
    #sum over the counts in centrality intervals# to save the plots
    recreate=False
    if index_pt is 0 and index_cen is 0:
      recreate=True
    Counts=Counts/Eff_BDT[index_pt*len(Centrality_bins)+index_cen]
    if index_cen == 3 and index_pt == 3:
      Fit_counts[index_cen].append([0,0])
    else:
      Fit_counts[index_cen].append(au.fit(Counts,Pt_bins[index_pt][0],Pt_bins[index_pt][1],recreate=recreate,filename=file_name))
    
    index_cut=index_cut+1
  
#loop to compute the efficiency
Effp = []
for index in range(0,len(Pt_bins)):
  Effp.append(Analysis.EfficiencyPresel(ct_cut=[0,100],pt_cut=Pt_bins[index],centrality_cut=[0,100]))

pt_binning = array("d",[0,2,4,6,8,10])
results = TFile(os.environ['HYPERML_DATA']+"/"+file_name,"update")

results.cd()

for index_cen in range(0,len(Fit_counts)):

  cv = TCanvas("pt_{}_{}".format(Centrality_bins[index_cen][0],Centrality_bins[index_cen][1]))
  histopt = TH1D("histopt_{}_{}".format(Centrality_bins[index_cen][0],Centrality_bins[index_cen][1]),";pT [GeV/c];dN/dpT [{GeV/c}^{-1}]",len(pt_binning)-1,pt_binning)


  gStyle.SetOptStat(0)
  gStyle.SetOptFit(0)
    ####################

  histopt.UseCurrentStyle()
  histopt.SetLineColor(1)
  histopt.SetMarkerStyle(20)
  histopt.SetMarkerColor(1)


  histopt.SetBinContent(1,0)
  histopt.SetBinError(1,0)
  for index in range(0,len(Fit_counts[index_cen])-1):
    histopt.SetBinContent(index+1,Fit_counts[index_cen][index][0]/Effp[index]/(pt_binning[index+1]-pt_binning[index]))
    histopt.SetBinError(index+1,Fit_counts[index_cen][index][1]/Effp[index]/(pt_binning[index+1]-pt_binning[index]))



  pinfo2= TPaveText(0.5,0.5,0.91,0.9,"NDC")
  pinfo2.SetBorderSize(0)
  pinfo2.SetFillStyle(0)
  pinfo2.SetTextAlign(30+3)
  pinfo2.SetTextFont(42)
  string ='ALICE Internal, Pb-Pb 2018 {}-{}%'.format(Centrality_bins[index_cen][0],Centrality_bins[index_cen][0])
  pinfo2.AddText(string)
  histopt.Draw()
  pinfo2.Draw("SAME")
    
  histopt.Write()
  cv.Write()
results.Close()

