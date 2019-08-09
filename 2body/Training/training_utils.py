# this class has been created to generalize the training and to open the file.root just one time
# to achive that alse analysis_utils.py and Significance_Test.py has been modified

#TODO: 
# ROC 
# wrong df in SignificanceScan
import uproot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pickle
from scipy.stats import norm
from scipy import stats
import pickle
#------------------------------
import analysis_utils as au
import Significance_Test as ST
import os

class Generalized_Analysis:

  def __init__(self,MCfile_name,Datafile_name,cut_presel,bkg_selection):

    centrality=uproot.open('../../../HypertritonData/EventCounter.root')['fCentrality']
    self.Centrality = [[0,10],[10,30],[30,50],[50,90]]
    self.n_ev = [0,0,0,0]
    
    for index in range(1,len(centrality)):
      if index<=self.Centrality[0][1]:
        self.n_ev[0]=centrality[index]+self.n_ev[0]
      elif index<=self.Centrality[1][1]:
        self.n_ev[1]=centrality[index]+self.n_ev[1]
      elif index<=self.Centrality[2][1]:
        self.n_ev[2]=centrality[index]+self.n_ev[2]
      elif index<=self.Centrality[3][1]:
        self.n_ev[3]=centrality[index]+self.n_ev[3]

    self.dfMCSig = uproot.open(MCfile_name)['SignalTable'].pandas.df()
    self.dfMCGen = uproot.open(MCfile_name)['GenTable'].pandas.df()
    self.dfData = uproot.open(Datafile_name)['DataTable'].pandas.df()

    self.dfData['Ct']=self.dfData['DistOverP']*2.99131
    self.dfMCSig['Ct']=self.dfMCSig['DistOverP']*2.99131

    self.dfMCSig['y'] = 1
    self.dfData['y'] = 0
    # dataframe for the background
    self.dfDataF = self.dfData.query(bkg_selection)
    # dataframe for the signal where are applied the preselection cuts
    self.dfMCSigF = self.dfMCSig.query(cut_presel)
    
    
  # function to compute the efficiency (it is not really useful)
  def Efficiency(self,variable,other_name,MaxRange,Step,plot = True, log = False):

    #plot of the all
    RecoHist,RecoDiv=np.histogram(self.dfMCSig[variable].array,bins=np.arange(0,MaxRange,Step))
    MCHist,MCdiv=np.histogram(self.dfMCGen[other_name].array,bins=np.arange(0,MaxRange,Step))
    #computation of the efficiency
    Eff = RecoHist/MCHist
    #binomial error 
    ErrEff = np.sqrt(Eff*(1-Eff)/MCHist)
    RecoDiv=RecoDiv+Step/2

    if plot is True:
      fig, ax = plt.subplots()
      ax.errorbar(RecoDiv[:-1], Eff,xerr=Step/2,yerr=ErrEff,linewidth=0.8)
      if log is True:
        ax.set_yscale('log')
      plt.ylabel("Efficiency")
      if variable is 'Ct':
        plt.title("Efficiency vs ct")  
        plt.xlabel("ct [cm]")
      elif variable is 'V0pt':
        plt.title("Efficiency vs pT")  
        plt.xlabel("pT [GeV/c]")
      plt.show()
    return (Eff,RecoDiv)
  # function to compute the preselection cuts efficiency
  def EfficiencyPresel(self,ct_cut=[0,100],pt_cut=[0,12],centrality_cut=[0,100]):
    ct_min = ct_cut[0]
    ct_max = ct_cut[1]
    pt_max = pt_cut[1]
    pt_min = pt_cut[0]
    centrality_max = centrality_cut[1]
    centrality_min = centrality_cut[0]
    total_cut = '@ct_min<Ct<@ct_max and @pt_min<V0pt<@pt_max and @centrality_min<Centrality<@centrality_max'
    total_cut_gen = '@ct_min<Ct<@ct_max and @pt_min<Pt<@pt_max and @centrality_min<Centrality<@centrality_max'
    countcut  = len(self.dfMCSigF.query(total_cut))/len(self.dfMCSig.query(total_cut))
    count = len(self.dfMCSig.query(total_cut))/len(self.dfMCGen.query(total_cut_gen))
    return countcut/count
  
  def TrainingAndTest(self,training_columns,params_def,ct_cut=[0,100],pt_cut=[2,3],centrality_cut=[0,10],num_rounds=200,draw=True,ROC=True):
    ct_min = ct_cut[0]
    ct_max = ct_cut[1]
    pt_max = pt_cut[1]
    pt_min = pt_cut[0]
    centrality_max = centrality_cut[1]
    centrality_min = centrality_cut[0]
    total_cut = '@ct_min<Ct<@ct_max and @pt_min<V0pt<@pt_max and @centrality_min<Centrality<@centrality_max'
    bkg = self.dfDataF.query(total_cut)
    sig = self.dfMCSigF.query(total_cut)
    print('condidates of bkg: ',len(bkg))
    print('condidates of sig: ',len(sig))
    if len(sig) is 0:
      print('no signal -> the model is not trained')
      return 0
    df= pd.concat([sig,bkg])
    traindata,testdata,ytrain,ytest = train_test_split(df[training_columns], df['y'], test_size=0.5)
    dtrain = xgb.DMatrix(data=np.asarray(traindata), label=ytrain, feature_names=training_columns)
    model = xgb.train(params_def, dtrain,num_boost_round=num_rounds)
    au.plot_output_train_test(model, traindata[training_columns], ytrain, testdata[training_columns], ytest, branch_names=training_columns,raw=True,log=True,draw=draw,ct_cut=ct_cut,pt_cut=pt_cut,centrality_cut=centrality_cut)
    # droc = xgb.DMatrix(data=np.asarray(testdata))
    # y_pred=model.predict(droc)
    # if ROC is True:
    #   au.plot_roc(ytest,y_pred)
    self.traindata = traindata
    self.testdata =testdata
    self.ytrain = ytrain
    self.ytest = ytest
    return model#,traindata,testdata,ytrain,ytest)

  #this function is still not working
  def Significance(self,model,training_columns,ct_cut=[0,100],pt_cut=[2,3],centrality_cut=[0,10]):

    ct_min = ct_cut[0]
    ct_max = ct_cut[1]
    pt_max = pt_cut[1]
    pt_min = pt_cut[0]
    centrality_max = centrality_cut[1]
    centrality_min = centrality_cut[0]
    total_cut = '@ct_min<Ct<@ct_max and @pt_min<V0pt<@pt_max and @centrality_min<Centrality<@centrality_max'
    dfDataSig=self.dfData.query(total_cut)

    dtest = xgb.DMatrix(data=(dfDataSig[training_columns]))
    y_pred = model.predict(dtest,output_margin=True)
    dfDataSig.eval('Score = @y_pred',inplace=True)    
    efficiency_array=au.EfficiencyVsCuts(dfDataSig)

    pT_list = [[2,3],[3,4],[4,5],[5,9]]
    i_pT = 0
    i_cen = 0
    for index in range(0,len(pT_list)):
      if pt_cut is pT_list[index]:
        print(index)
        i_pT=index
        break
    for index in range(0,len(self.Centrality)):
      if centrality_cut is self.Centrality[index]:
        i_cen=index
        break

    return ST.SignificanceScan(dfDataSig,ct_cut,pt_cut,centrality_cut,i_pT,efficiency_array,self.EfficiencyPresel(ct_cut,pt_cut,centrality_cut),self.n_ev[i_cen])



