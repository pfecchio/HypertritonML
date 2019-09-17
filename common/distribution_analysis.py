import os
import warnings
import argparse
import yaml
import numpy as np
from array import array
from ROOT import TF1, TH1D, TH2D, TCanvas, TFile, TPaveText, gDirectory, gStyle ,gROOT , TIter, TKey, TClass

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

file_name = resultsSysDir +  '/' + params['FILE_PREFIX'] + '_dist.root'
distribution = TFile(file_name,'recreate')

file_name = resultsSysDir +  '/' + params['FILE_PREFIX'] + '_results.root'
resultFile = TFile(file_name)



for cclass in params['CENTRALITY_CLASS']:
  dir_name =  "{}-{}".format(cclass[0],cclass[1])
  resultFile.cd(dir_name)
  list_of_object = [key.GetName() for key in gDirectory.GetListOfKeys()]

  histo_counts = []
  histo_bdt = []

  if args.pt:
    histo_sel_eff = gROOT.FindObject("SelEff").ProjectionX()
  else:
    histo_sel_eff = gROOT.FindObject("SelEff").ProjectionY()

  for object_name in list_of_object:
    if object_name[:9]=='RawCounts':
      if args.pt:
        histo = gROOT.FindObject(object_name).ProjectionX()
        histo_bdt_eff = gROOT.FindObject("BDTeff"+object_name[9:]).ProjectionX()
      else:
        histo = gROOT.FindObject(object_name).ProjectionY()
        histo_bdt_eff = gROOT.FindObject("BDTeff"+object_name[9:]).ProjectionY()
      
      histo.SetName('histo_'+object_name[9:])
      histo_counts.append(histo)
      histo_bdt.append(histo_bdt_eff)

  tau = []
  err_tau = []
  N0 = []
  err_N0 = []
  chi2_over_ndf = []
  err_chi2_over_ndf = []
  for histo in histo_counts:
    for bin in range(1,histo.GetNbinsX()+1):
      eff = histo_sel_eff.GetBinContent(bin)*histo_bdt_eff.GetBinContent(bin)
      
      bin_width = histo.GetBinWidth(bin)
      histo.SetBinContent(bin,histo.GetBinContent(bin)/eff/bin_width)
      histo.SetBinError(bin,histo.GetBinError(bin)/eff/bin_width)
      histo.GetYaxis().SetTitle('dN/dct [cm^{-1}]')
    distribution.cd()

    #this part can be skipped

    cv = TCanvas("ct_"+histo.GetName()[5:])
    cv.SetLogy()
    gStyle.SetOptStat(0)
    gStyle.SetOptFit(0)

    histo.UseCurrentStyle()
    histo.SetLineColor(1)
    histo.SetMarkerStyle(20)
    histo.SetMarkerColor(1)

    expo = TF1("","[0]*exp(-x/[1]/0.029979245800)",0,28)
    expo.SetParLimits(1,100,350)
    histo.Fit(expo,"MIR")

    pinfo2= TPaveText(0.5,0.5,0.91,0.9,"NDC")
    pinfo2.SetBorderSize(0)
    pinfo2.SetFillStyle(0)
    pinfo2.SetTextAlign(30+3)
    pinfo2.SetTextFont(42)
    string ='ALICE Internal, Pb-Pb 2018 {}-{}%'.format(0,90)
    pinfo2.AddText(string)
    string='#tau = {:.0f} #pm {:.0f} ps '.format(expo.GetParameter(1),expo.GetParError(1))
    pinfo2.AddText(string)  
    if expo.GetNDF()is not 0:
      string='#chi^{{2}} / NDF = {}'.format(expo.GetChisquare() / (expo.GetNDF()))
    pinfo2.AddText(string)  
    pinfo2.Draw()
    
    tau.append(expo.GetParameter(1))
    err_tau.append(expo.GetParError(1))
    N0.append(expo.GetParameter(0))
    err_N0.append(expo.GetParError(0))
    if expo.GetNDF()is not 0:
      chi2_over_ndf.append(expo.GetChisquare() / (expo.GetNDF()))
    else:
      chi2_over_ndf.append(0)


    histo.Write()
    cv.Write()

  tau.remove(tau[0])
  err_tau.remove(err_tau[0])
  N0.remove(N0[0])
  err_N0.remove(err_N0[0])
  chi2_over_ndf.remove(chi2_over_ndf[0])

  params['BDT_EFFICIENCY'].append(params['BDT_EFFICIENCY'][len(params['BDT_EFFICIENCY'])-1])
  params['BDT_EFFICIENCY'].reverse()
  params['BDT_EFFICIENCY'].append(params['BDT_EFFICIENCY'][len(params['BDT_EFFICIENCY'])-1])

  distribution.cd()
  histo_tau = TH1D("histo_tau",";efficiency;#tau [ps]",len(histo_counts),array('d',params['BDT_EFFICIENCY']))
  for index in range(1,len(histo_counts)-1):
    histo_tau.SetBinContent(len(histo_counts)-index+1,tau[index])
    histo_tau.SetBinError(len(histo_counts)-index+1,err_tau[index])
  print('std tau :',np.std(tau))

  histo_n0 = TH1D("histo_n0",";efficiency;counts",len(histo_counts),array('d',params['BDT_EFFICIENCY']))
  for index in range(1,len(histo_counts)-1):
    histo_n0.SetBinContent(len(histo_counts)-index+1,N0[index])
    histo_n0.SetBinError(len(histo_counts)-index+1,err_N0[index])
  print('std n0 :',np.std(N0))

  histo_chi2 = TH1D("histo_chi2",";efficiency;chi^{2}/ndf",len(histo_counts),array('d',params['BDT_EFFICIENCY']))
  for index in range(1,len(histo_counts)-1):
    histo_chi2.SetBinContent(len(histo_counts)-index+1,chi2_over_ndf[index])
    histo_chi2.SetBinError(len(histo_counts)-index+1,0)

  histo_tau.Write()
  histo_n0.Write()
  histo_chi2.Write()

resultFile.Close()






