import os
import warnings
import argparse
import yaml
import numpy as np
from array import array
from ROOT import TF1, TH1D, TH2D, TCanvas, TFile, TPaveText, gDirectory, gStyle ,gROOT , TIter, TKey, TClass, gRandom,TTree
import math
gROOT.SetBatch()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pt", help="compute pt distribution otherwise ct distribution", action="store_true")
parser.add_argument("config", help="Path to the YAML configuration file")
args = parser.parse_args()

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

score_bdteff_name = resultsSysDir + '/{}_score_bdteff.yaml'.format(params['FILE_PREFIX'])
with open(os.path.expandvars(score_bdteff_name), 'r') as stream:
    try:
        data = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

file_name = resultsSysDir +  '/' + params['FILE_PREFIX'] + '_results.root'
resultFile = TFile(file_name)

file_name = resultsSysDir +  '/' + params['FILE_PREFIX'] + '_syst.root'
distribution = TFile(file_name,'recreate')

tree = TTree( 'distr_data', 'tree with the information about the distributions' )

tau_Tree = np.array([0], dtype=np.float32)
err_tau_Tree = np.array([0], dtype=np.float32)
prob = np.array([0], dtype=np.float32)
count_Tree = np.array([0,0,0,0,0,0,0,0,0], dtype=np.float32)
err_count_Tree = np.array([0,0,0,0,0,0,0,0,0], dtype=np.float32 )
eff_Tree = np.array([0,0,0,0,0,0,0,0,0], dtype=np.float32 )
z_Tree = np.array([0], dtype=np.float32 )

tree.Branch( 'tau', tau_Tree, 'tau/F' )
tree.Branch( 'err_tau', err_tau_Tree, 'err_tau/F' )
tree.Branch( 'prob', prob, 'prob/F' )
tree.Branch( 'counts', count_Tree, 'counts[9]/F' )
tree.Branch( 'err_counts', err_count_Tree, 'err_counts[9]/F' )
tree.Branch( 'eff', eff_Tree, 'eff[9]/F' )
tree.Branch( 'z', z_Tree, 'z/F' )

for cclass in params['CENTRALITY_CLASS']:
  dir_name =  "{}-{}".format(cclass[0],cclass[1])
  resultFile.cd(dir_name)
  list_of_object = [key.GetName() for key in gDirectory.GetListOfKeys()]

  histo_counts = []
  histo_bdt = []

  if args.pt:
    histo = gROOT.FindObject('RawCounts').ProjectionX()
    histo_presel_eff = gROOT.FindObject("SelEff").ProjectionX()
  else:
    histo = gROOT.FindObject('RawCounts').ProjectionY()
    histo_presel_eff = gROOT.FindObject("SelEff").ProjectionY()


  histo_bdt_eff = histo.Clone()
  for bin in range(1,histo.GetNbinsX()+1):
    if args.pt:
      first_index = 'CENT[{}, {}]_PT({}, {})_CT({}, {})'.format(cclass[0],cclass[1],params['PT_BINS'][bin-1],params['PT_BINS'][bin],params['CT_BINS'][0],params['CT_BINS'][1])
    else:  
      first_index = 'CENT[{}, {}]_PT({}, {})_CT({}, {})'.format(cclass[0],cclass[1],params['PT_BINS'][0],params['PT_BINS'][1],params['CT_BINS'][bin-1],params['CT_BINS'][bin])
        
    histo_bdt_eff.SetBinContent(bin,data[first_index]['sig_scan'][1])

  for bin in range(1,histo.GetNbinsX()+1):
    eff = histo_presel_eff.GetBinContent(bin)*histo_bdt_eff.GetBinContent(bin)
    bin_width = histo.GetBinWidth(bin)
    histo.SetBinContent(bin,histo.GetBinContent(bin)/eff/bin_width)
    histo.SetBinError(bin,histo.GetBinError(bin)/eff/bin_width)

  expo = TF1("","[0]*exp(-x/[1]/0.029979245800)",2,23)
  expo.SetParLimits(1,100,350)                      
  histo.Fit(expo,"MIR")
  tau_max = expo.GetParameter(1)
  err_tau_max = expo.GetParError(1)
  histo_syst = TH1D('distri_syst',';z;counts',100,-20,20)
  histo_syst_f = TH1D('distri_syst_good_fit',';z;counts',100,-20,20)
  histo_syst_f2 = TH1D('distri_syst_good_fit2',';z;counts',100,-20,20)
  histo_syst_f3 = TH1D('distri_syst_good_fit2',';z;counts',100,-20,20)
  gRandom.SetSeed(0)
  index_fit = 0
 
  histoct = TH1D('histoct','',len(params['CT_BINS'])-1,array('d',params['CT_BINS']))
  
  for n in range(0,10000):

    efficiency = []
    for index in range(0,9):
      efficiency.append(params['BDT_EFFICIENCY'][int(gRandom.Rndm()*len(params['BDT_EFFICIENCY']))])
    print(efficiency)
    counts = []
    ct_bin = 1

    for eff,ctmin,ctmax in zip(efficiency,params['CT_BINS'][:-1], params['CT_BINS'][1:]):
      first_index ='CENT[{}, {}]_PT(2, 10)_CT({}, {})'.format(cclass[0],cclass[1],ctmin,ctmax)
      count = data[first_index]['eff{}'.format(eff)][2]/data[first_index]['eff{}'.format(eff)][1]/histo_presel_eff.GetBinContent(ct_bin)
      count_err = data[first_index]['eff{}'.format(eff)][3]/data[first_index]['eff{}'.format(eff)][1]/histo_presel_eff.GetBinContent(ct_bin)
      histoct.SetBinContent(index,count)
      histoct.SetBinError(ct_bin,count_err)
      counts.append(count)
      eff_Tree[ct_bin-1] = eff
      count_Tree[ct_bin-1] = count
      err_count_Tree[ct_bin-1] = count_err
      ct_bin = ct_bin+1
      
    expo.SetParLimits(1,150,350)   
    histoct.Fit(expo,"MRQ")

    index_fit = index_fit+1
    if  err_tau_max-expo.GetParError(1)!=0 and expo.GetParameter(1)<350 and expo.GetParameter(1)>150:
      z = (tau_max-expo.GetParameter(1))/(math.fabs(err_tau_max-expo.GetParError(1)))
      histo_syst_f.Fill(z)
    if  err_tau_max-expo.GetParError(1)!=0 and expo.GetParameter(1)<300 and expo.GetParameter(1)>170:
      z = (tau_max-expo.GetParameter(1))/(math.fabs(err_tau_max-expo.GetParError(1)))
      histo_syst_f2.Fill(z)
    if  err_tau_max-expo.GetParError(1)!=0 and expo.GetParameter(1)<300 and expo.GetParameter(1)>190:
      z = (tau_max-expo.GetParameter(1))/(math.fabs(err_tau_max-expo.GetParError(1)))
      histo_syst_f3.Fill(z)
    if err_tau_max-expo.GetParError(1)!=0:
      z = (tau_max-expo.GetParameter(1))/(math.fabs(err_tau_max-expo.GetParError(1)))
      histo_syst.Fill(z)
      tau_Tree[0] = expo.GetParameter(1)
      err_tau_Tree[0] = expo.GetParError(1)
      prob[0] = expo.GetProb()
      z_Tree[0] = z
      tree.Fill()

  distribution.cd()
  histo_syst.Write()
  histo_syst_f.Write()
  histo_syst_f2.Write()
  histo_syst_f3.Write()
  tree.Write()
  distribution.Close()

resultFile.Close()
