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
from generalized_analysis import GeneralizedAnalysis
import warnings


# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

def ct_analysis(training_columns,params_def,Training = False,Significance=False,custom=False,score_shift=0,file_name='results.root'):

  Analysis = GeneralizedAnalysis(2,os.environ['HYPERML_TABLES_2']+'/SignalTable.root',os.environ['HYPERML_TABLES_2']+'/DataTable.root','2<=HypCandPt<=10','(InvMass<2.98 or InvMass>3.005) and HypCandPt<=10')
  #Analysis_bkg = GeneralizedAnalysis(2,os.environ['HYPERML_TABLES_2']+'/SignalTable.root',os.environ['HYPERML_TABLES_2']+'/DataTable_bkg.root','2<=HypCandPt<=10','(InvMass<2.98 or InvMass>3.005) and HypCandPt<=10')

  #Analysis.correlation_plot(training_columns,draw=True)
  
  # loop to train the models
  if not os.path.exists(os.environ['HYPERML_MODELS_2']):
    os.makedirs(os.environ['HYPERML_MODELS_2'])
  Centrality_bins = [[0,10],[10,30],[30,50],[50,90]]
  Ct_bins = [[0,2],[2,4],[4,6],[6,8],[8,10],[10,14],[14,18],[18,23],[23,28]]
  
  # nev_MC = []
  Cut_saved = []
  Eff_BDT = []
  for index_ct in range(0,len(Ct_bins)):
    for index_cen in range(0,len(Centrality_bins)):   
      print('centrality: ',Centrality_bins[index_cen])
      print('Ct: ',Ct_bins[index_ct])

      if Training is True or Significance is True:
        data = Analysis.prepare_dataframe(training_columns, cent_class=Centrality_bins[index_cen], ct_range=Ct_bins[index_ct],bkg_factor=10)
      
          
      filename = os.environ['HYPERML_MODELS_2']+'/BDT_Ct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_Cen_{:.2f}_{:.2f}.sav'.format(Ct_bins[index_ct][0],Ct_bins[index_ct][1],0,10,Centrality_bins[index_cen][0],Centrality_bins[index_cen][1])
      if Training is True:
        print('training the models ...')
        model =Analysis.train_test_model(data, training_columns, params_def, Ct_bins[index_ct],[0,10],Centrality_bins[index_cen],optimize=False)
        # nev_MC.append(Analysis.number_events_MC(Ct_bins[index_ct],[0,10],Centrality_bins[index_cen]))
        if model is not 0:
          pickle.dump(model, open(filename, 'wb'))
          print(filename+' has been saved')
      
      if Significance is True:
        print('computing the cuts ...')
        model = pickle.load(open(filename, 'rb'))
        Cut,eff = Analysis.significance_scan(data,model,training_columns,ct_range=Ct_bins[index_ct],pt_range=[0,10],cent_class=Centrality_bins[index_cen],custom=custom)
        Cut_saved.append(Cut)
        Eff_BDT.append(eff)
  
  # if Training is False:
  #   nev_MC = [17036.0, 81426.0, 61730.0, 69320.0, 18758.0, 86913.0, 66072.0, 74971.0, 14928.0, 68138.0, 51422.0, 58989.0, 12171.0, 53709.0, 40338.0, 45018.0, 9747.0, 43213.0, 32366.0, 35994.0, 13860.0, 61172.0, 45976.0, 51552.0, 8795.0, 38118.0, 28780.0, 32175.0, 6535.0, 28367.0, 20998.0, 23548.0, 3617.0, 15522.0, 11554.0, 12734.0]
  if Significance is False:
    Eff_BDT = [0.5234218289085546, 0.6777731636416934, 0.7574905459129254, 0.7884154960048461, 0.7386999569522169, 0.8640734366035571, 0.8986059451449998, 0.89413771960225, 0.7750755702115966, 0.8742737014219393, 0.908104309806371, 0.907210902803675, 0.7820428336079077, 0.8741094408594129, 0.9110845295055822, 0.9160441124155105, 0.7883378294251919, 0.8538539929590513, 0.8768049155145929, 0.8684430105134339, 0.7215560105293946, 0.809695551873152, 0.8176619521564519, 0.8302690149022643, 0.65860400829302, 0.7677959850360925, 0.8209389281262983, 0.5500030885168942, 0.5914747977598008, 0.6395381385584324, 0.7511467889908257, 0.8132053225534782, 0.3521613832853026, 0.6373136230406525, 0.6005835907998627, 0.7173166926677067]
    Cut_saved=  [3.2, 5.1000000000000005, 5.5, 7.9, 3.7, 5.5, 5.800000000000001, 8.0, 3.8000000000000007, 5.800000000000001, 5.9, 7.800000000000001, 3.8000000000000007, 5.800000000000001, 5.800000000000001, 7.600000000000001, 4.0, 6.1, 6.300000000000001, 7.800000000000001, 4.4, 6.4, 6.800000000000001, 8.100000000000001, 4.4, 6.5, 6.5, 9.200000000000001, 4.5, 6.9, 6.6, 7.700000000000001, 5.300000000000001, 6.4, 7.1, 7.700000000000001]

  print("efficiency BDT: ",Eff_BDT)
  print("cut: ",Cut_saved)
  # print("nevent: ",nev_MC)
  Fit_counts = []
  # loop to read the models and to do the prediction
  index_cut = 0
  plt.close()
  if not os.path.exists(os.environ['HYPERML_FIGURES_2']+'/Peaks/'):
    os.makedirs(os.environ['HYPERML_FIGURES_2']+'/Peaks/')
  for index_ct in range(0,len(Ct_bins)):
    for index_cen in range(0,len(Centrality_bins)):
      output_cut =  Cut_saved[index_cut]
      print('centrality: ',Centrality_bins[index_cen],'Ct: ',Ct_bins[index_ct])
      
      ct_min = Ct_bins[index_ct][0]
      ct_max = Ct_bins[index_ct][1]
      centrality_min = Centrality_bins[index_cen][0]
      centrality_max = Centrality_bins[index_cen][1]

      total_cut = '@ct_min<ct<@ct_max and 0<HypCandPt<10 and @centrality_min<centrality<@centrality_max'
      filename = '/BDT_Ct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_Cen_{:.2f}_{:.2f}.sav'.format(Ct_bins[index_ct][0],Ct_bins[index_ct][1],0,10,Centrality_bins[index_cen][0],Centrality_bins[index_cen][1])
      ##per testare il dev
      #filename = '/BDT_{}{}_{}{}_{}{}.sav'.format(Centrality_bins[index_cen][0],Centrality_bins[index_cen][1],0,10,Ct_bins[index_ct][0],Ct_bins[index_ct][1])
      model = pickle.load(open(os.environ['HYPERML_MODELS_2']+filename, 'rb'))
      dfDataF = Analysis.df_data_all.query(total_cut)
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
    Fit_counts.append(au.fit(CountsTot,Ct_bins[index_ct][0],Ct_bins[index_ct][1],recreate=recreate,filename=file_name))
    

  #loop to compute the efficiency
  Effp = []
  for index in range(0,len(Ct_bins)):
    Effp.append(Analysis.preselection_efficiency(ct_range=Ct_bins[index],pt_range=[0,10],cent_class=[0,100]))

  ct_binning = array("d",[0,2,4,6,8,10,14,18,23,28])
  results = TFile(os.environ['HYPERML_DATA_2']+"/"+file_name,"update")

  results.cd()

  #total efficiency
  for index_cen in range(0,len(Centrality_bins)):
    histo_eff = TH1D("histo_eff_{}_{}".format(Centrality_bins[index_cen][0],Centrality_bins[index_cen][1]),";ct [cm];efficiency",len(ct_binning)-1,ct_binning)
    for index_ct in range(0,len(Ct_bins)):
      errBDT = math.sqrt((1-Effp[index_ct])*Effp[index_ct])
      errEff = math.sqrt((1-Eff_BDT[index_cen*4+index_ct])*Eff_BDT[index_cen*4+index_ct])
      # errTot = Eff_BDT[index_ct+index_cen*4]*Effp[index_ct]*math.sqrt((errBDT*errBDT/Eff_BDT[index_ct+index_cen*4]/Eff_BDT[index_ct+index_cen*4]+errEff*errEff/Effp[index_ct]/Effp[index_ct])/nev_MC[index_cen*4+index_ct])
      errTot = 0
      histo_eff.SetBinContent(index_ct+1,Effp[index_ct]*Eff_BDT[index_cen*4+index_ct])
      histo_eff.SetBinError(index_ct+1,errTot)
    histo_eff.Write()

  #ct distribution
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
  expo.SetParLimits(1,180,280)
  histoct.Fit(expo,"M")

  pinfo2= TPaveText(0.5,0.5,0.91,0.9,"NDC")
  pinfo2.SetBorderSize(0)
  pinfo2.SetFillStyle(0)
  pinfo2.SetTextAlign(30+3)
  pinfo2.SetTextFont(42)
  string ='ALICE Internal, Pb-Pb 2018 {}-{}%'.format(0,90)
  pinfo2.AddText(string)
  string='#tau = {:.0f} #pm {:.0f} ps '.format(expo.GetParameter(1),expo.GetParError(1))
  pinfo2.AddText(string)  
  string='prob = {}'.format(expo.GetProb())
  pinfo2.AddText(string)  
  pinfo2.Draw()
    
  histoct.Write()
  cv.Write()
  results.Close()
  return (expo.GetParameter(1),expo.GetParError(1))



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

training_columns = [[ 'V0CosPA','ProngsDCA','PiProngPvDCAXY','He3ProngPvDCAXY','HypCandPt','He3ProngPvDCA','PiProngPvDCA','NpidClustersHe3','TPCnSigmaHe3'],
[ 'PiProngPvDCAXY','He3ProngPvDCAXY','He3ProngPvDCA','PiProngPvDCA','NpidClustersHe3','TPCnSigmaHe3'],
[ 'V0CosPA','ProngsDCA','HypCandPt','ArmenterosAlpha','NpidClustersHe3','TPCnSigmaHe3'],
[ 'V0CosPA','ProngsDCA','HypCandPt','ArmenterosAlpha','PiProngPvDCAXY','He3ProngPvDCAXY','He3ProngPvDCA','PiProngPvDCA']]

ct_analysis(training_columns[0],params_def,Training=False,Significance=True,score_shift=0,custom=True,file_name='/results_ct.root')

# syst_file = TFile(os.environ['HYPERML_DATA']+"/systematic_ct.root","recreate")
# histo_shift = TH1D("histo_shift",";cut-score;#tau [ps]",5,-1,1)
# tau_shift = []
# err_tau_shift = []
# bin = 1
# for shift in np.linspace(-1,1,5):
#  print(shift)
#  tau,err = ct_analysis(training_columns[0],params_def,Training=False,Significance=True,score_shift=shift,file_name='/results_'+str(shift)+'.root')
#  tau_shift.append(tau)
#  err_tau_shift.append(err)
#  histo_shift.SetBinContent(bin,tau_shift[bin-1])
#  histo_shift.SetBinError(bin,err_tau_shift[bin-1])
#  bin=bin+1
# print('syst shift :',np.std(tau_shift))

# histo_col = TH1D("histo_col",";;",len(training_columns),0,len(training_columns))
# tau_col = []
# err_tau_col = []
# for index in range(0,len(training_columns)):
#   tau,err = ct_analysis(training_columns[index],params_def,Training=True,score_shift=shift,file_name='/results_col_'+str(index)+'.root')
#   tau_col.append(tau)
#   err_tau_col.append(err)
#   histo_col.SetBinContent(index,tau_col[index])
#   histo_col.SetBinError(index,err_tau_col[index])
# print('syst col :',np.std(tau_col[0]))

# histo_col.Write()
# syst_file.cd()
# histo_shift.Write()
# syst_file.Close()
