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
from ROOT import TH2D,TFile

def Write_List(file_name,numbers,title=''):
  file = open(file_name,'a+')
  final_string = ''
  for item in numbers:
    final_string = final_string+str(item)+' '
  file.write(title)
  file.write(final_string+'\n')
  file.close()

file_name = os.environ['HYPERML_FIGURES']+'/results.txt'
# file = open(file_name,'a+')
# file.close()

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
params_def = {'max_depth': 8, 'eta': 0.1, 'gamma': 0.2, 'min_child_weight': 2, 'subsample': 0.6, 'colsample_bytree': 0.9, 'objective': 'binary:logistic', 'random_state': 42, 'silent': 1, 'nthread': 4, 'tree_method': 'hist', 'scale_pos_weight': 10}

training_columns = [ 'V0CosPA','ProngsDCA', 'DistOverP','ArmenterosAlpha','NpidClustersHe3','V0pt','TPCnSigmaHe3','He3ProngPvDCA','PiProngPvDCA']

Analysis = tu.Generalized_Analysis(os.environ['HYPERML_TABLES']+'/SignalTable.root',os.environ['HYPERML_TABLES']+'/DataTable.root','ProngsDCA<1.6 and He3ProngPvDCA>0.01 and He3ProngPvDCA>0.01 and V0CosPA>0.98','(InvMass<2.98 or InvMass>3.005) and V0pt<=10')

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
    model =Analysis.TrainingAndTest(training_columns,params_def,Ct_bins[index_ct],[2,10],Centrality_bins[index_cen],draw=False)
    if model is not 0:
      filename = '/BDT_Ct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_Cen_{:.2f}_{:.2f}.sav'.format(Ct_bins[index_ct][0],Ct_bins[index_ct][1],0,10,Centrality_bins[index_cen][0],Centrality_bins[index_cen][1])
      pickle.dump(model, open(os.environ['HYPERML_MODELS']+filename, 'wb'))
      Cut,eff = Analysis.Significance(model,training_columns,Ct_bins[index_ct],[2,10],Centrality_bins[index_cen],draw=False)
      Cut_saved.append(Cut)
      Eff_BDT.append(eff)
      print(filename+' has been saved')

print("efficiency BDT: ",Eff_BDT)
print("cut: ",Cut_saved)


# cuts obtained by the previous loop
# Cut_saved = [4.202020202020202, 4.515151515151516, 5.767676767676768, 3.7323232323232327, 4.358585858585859, 5.297979797979799, 5.767676767676768, 4.045454545454546, 4.515151515151516, 4.984848484848485, 6.080808080808081, 3.262626262626263, 4.828282828282829, 5.141414141414142, 6.080808080808081, 2.47979797979798, 4.671717171717172, 4.984848484848485, 5.767676767676768, 1.070707070707071, 5.141414141414142, 5.611111111111111, 5.767676767676768, 4.358585858585859, 5.454545454545455, 5.611111111111111, 6.3939393939393945, 4.828282828282829, 5.141414141414142, 6.080808080808081, 6.707070707070708, 2.7929292929292933, 4.984848484848485, 5.611111111111111, 6.863636363636363, 4.358585858585859]

#

results = TFile(os.environ['HYPERML_DATA']+'/results.root','RECREATE')
histo = TH2D('InvMassVsct','histo;m [GeV/c^2];ct bin [cm];counts',30,2.96,3.05,9,0,9)

Ct_counts = []
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
    Counts,bins = np.histogram(dfDataF.query('Score >@output_cut')['InvMass'],bins=30,range=[2.96,3.05])
    #sum over the counts in centrality intervals
    if index_cen==0:
      CountsTot=Counts
    else:
      CountsTot=CountsTot+Counts
    index_cut=index_cut+1
  ##
  Write_List(file_name,CountsTot,str(Ct_bins[index_ct])+'\n')
  Write_List(file_name,bins)

  for index_mass in range(0,30,1):
    histo.SetBinContent(index_mass+1,index_ct+1,CountsTot[index_mass])
    histo.SetBinError(index_mass+1,index_ct+1,math.sqrt(CountsTot[index_mass]))
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
#Eff_BDT = [0.48680042238648363, 0.7269054303910155, 0.6885647103987715, 0.8886925795053003, 0.7239274502964772, 0.8271386627519017, 0.8405461698680861, 0.9182222222222223, 0.753260637160573, 0.8703726889528818, 0.8728208728208728, 0.9367755532139094, 0.756673373574782, 0.8698046275215757, 0.8282305267897343, 0.9388646288209607, 0.7888019060585433, 0.8802303005958358, 0.7924064979221761, 0.9283276450511946, 0.7277620061509344, 0.8308816505874487, 0.78334125959644, 0.8366834170854272, 0.6802064417471089, 0.7802648904035319, 0.7105843439911798, 0.7254098360655737, 0.615156209656597, 0.7201805963880722, 0.6209000762776506, 0.88, 0.5536929057337221, 0.6986881937436933, 0.5426621160409556, 0.6910994764397905]
#loop to compute the efficiency
Effp = []
for index in range(0,len(Ct_bins)):
  Effp.append(Analysis.EfficiencyPresel(Ct_bins[index],pt_cut=[0,10],centrality_cut=[0,100]))
  Ct_counts[index]=Ct_counts[index]/Effp[index]/Eff_BDT[index]
errCt = np.sqrt(Ct_counts)
print('eff presel: ',Effp)
def expo(x,n,tau):
    return n*np.exp(-x/tau/0.029979245800)


Write_List(file_name,Eff_BDT,'bdt\n')
Write_List(file_name,Effp,'presel\n')

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
