#!/usr/bin/env python3

import os
import warnings
import argparse
import yaml
import numpy as np
from array import array
from ROOT import TF1, TH1D, TH2D, TCanvas, TFile, TPaveText, gDirectory, gStyle ,gROOT , TIter, TKey, TAxis

gROOT.SetBatch()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pt", help="compute pt distribution otherwise ct distribution", action="store_true")
parser.add_argument("config", help="Path to the YAML configuration file")
args = parser.parse_args()

expo = TF1("","[0]*exp(-x/[1]/0.029979245800)",0,28)
expo.SetParLimits(1,100,350)

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
        eff_data = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

if args.pt:
  var = '#it{p}_{T}'
  unit = 'GeV/#it{c}'
else:
  var = '#it{ct}'
  unit = 'cm'

file_name = resultsSysDir +  '/' + params['FILE_PREFIX'] + '_dist.root'
distribution = TFile(file_name,'recreate')

file_name = resultsSysDir +  '/' + params['FILE_PREFIX'] + '_results.root'
resultFile = TFile(file_name)



for cclass in params['CENTRALITY_CLASS']:
  dir_name =  "{}-{}".format(cclass[0],cclass[1])
  resultFile.cd(dir_name)
  outDir = distribution.mkdir(dir_name)
  cvDir = outDir.mkdir("canvas")
  list_of_object = [key.GetName() for key in gDirectory.GetListOfKeys()]

  histo_counts = []
  histo_bdt = []

<<<<<<< HEAD
  tau = []
  err_tau = []
  bin_label = []

=======
  h2PreselEff = resultFile.Get("{}/SelEff".format(dir_name))
>>>>>>> Easier logic for the distribution analysis macro
  if args.pt:
    histo_presel_eff = h2PreselEff.ProjectionX()
  else:
    histo_presel_eff = h2PreselEff.ProjectionY()

  for i in range(1,histo_presel_eff.GetNbinsX()+1):
    histo_presel_eff.SetBinError(i,0)

  histo_presel_eff.SetTitle(";{} ({}); Preselection efficiency".format(var,unit))
  histo_presel_eff.UseCurrentStyle()

  outDir.cd()
  histo_presel_eff.Write("Preselection_Efficiency")

  bdt_effs = np.arange(0.5,0.71,0.01)
  results = dict()
  for fix_eff in  params['BDT_EFFICIENCY']:
    if (float(fix_eff) > 0.7):
      continue
    h2RawCounts = resultFile.Get('{}/RawCounts{}'.format(dir_name,fix_eff))
    h1RawCounts = h2RawCounts.ProjectionX() if args.pt else h2RawCounts.ProjectionY()
    h1RawCounts.SetTitle(";dN/d#it{ct} (cm)^{-1};#it{ct} (cm)")
    h1RawCounts.Divide(histo_presel_eff)
    h1RawCounts.Scale(1./float(fix_eff),'width')
    outDir.cd()
    h1RawCounts.UseCurrentStyle()
    h1RawCounts.Fit(expo,"MI")
    h1RawCounts.Write('ctSpectra{}'.format(fix_eff))

    cvDir.cd()
    myCv = TCanvas("ctSpectraCv{}".format(fix_eff))
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
    h1RawCounts.Draw()
    pinfo2.Draw()
    cvDir.cd()
    myCv.Write()
    results[fix_eff] = [expo.GetParameter(1),expo.GetParError(1)]

  delta = TH1D("delta",";n#sigma;Entries",30,-3,3)
  for key, value in  results.items():
    if key == 0.6:
      continue
    delta.Fill((value[0] - results[round(0.6,1)][0]) / results[round(0.6,1)][1])
  outDir.cd()
  delta.Write()

  


  # for object_name in list_of_object:
  #   if object_name[:9]=='RawCounts':
  #     if args.pt:
  #       histo = gROOT.FindObject(object_name).ProjectionX()
  #       histo.SetTitle(';p_{T} [ct];dN/dp_{T} [{GeV/c}^{-1}]')

  #     else:
  #       histo = gROOT.FindObject(object_name).ProjectionY()
  #       histo.SetTitle(';ct [cm];dN/dct [cm^{-1}]')

  #     #histo.SetTitle(';{} [{}];dN/d{} [{}^{-1}]'.format(var,unit,var,unit))
      

<<<<<<< HEAD
      if object_name=='RawCounts':
        second_index = 'sig_scan'
      else:
        second_index = 'eff'+object_name[9:]
      
      bin_label.append(second_index)


      histo_eff = histo.Clone()
      for bin in range(1,histo.GetNbinsX()+1):
        if args.pt:
          first_index = 'CENT[{}, {}]_PT({}, {})_CT({}, {})'.format(cclass[0],cclass[1],params['PT_BINS'][bin-1],params['PT_BINS'][bin],params['CT_BINS'][0],params['CT_BINS'][1])
        else:  
          first_index = 'CENT[{}, {}]_PT({}, {})_CT({}, {})'.format(cclass[0],cclass[1],params['PT_BINS'][0],params['PT_BINS'][1],params['CT_BINS'][bin-1],params['CT_BINS'][bin])
=======
  #     if object_name=='RawCounts':
  #       second_index = 'sig_scan'
  #     else:
  #       second_index = 'eff'+object_name[9:]

  #     histo_eff = histo.Clone()
  #     for bin in range(1,histo.GetNbinsX()+1):
  #       if args.pt:
  #         first_index = 'CENT[{}, {}]_PT({}, {})_CT({}, {})'.format(cclass[0],cclass[1],params['PT_BINS'][bin-1],params['PT_BINS'][bin],params['CT_BINS'][0],params['CT_BINS'][1])
  #       else:  
  #         first_index = 'CENT[{}, {}]_PT({}, {})_CT({}, {})'.format(cclass[0],cclass[1],params['PT_BINS'][0],params['PT_BINS'][1],params['CT_BINS'][bin-1],params['CT_BINS'][bin])
>>>>>>> Easier logic for the distribution analysis macro
        
          
  #       histo_eff.SetBinContent(bin,eff_data[first_index][second_index][1])
  #       histo_eff.SetName('eff'+object_name[9:])
  #     histo.SetName('histo_'+object_name[9:])
  #     histo_counts.append(histo)
  #     histo_bdt.append(histo_eff)

  # histograms for the BDT and total efficiency
  # eff_tot = histo_bdt[0].Clone()
  # eff_tot.SetName('eff_tot')

  # eff_bdt = histo_bdt[0].Clone()
  # eff_bdt.SetName('eff_bdt')
  
  # eff_tot.SetTitle(';{} [{}]; total efficiency'.format(var,unit))
  # eff_bdt.SetTitle(';{} [{}]; BDT efficiency '.format(var,unit))
  # histo_presel_eff.SetTitle(';{} [{}]; preselection efficiency '.format(var,unit))

  # for bin in range(1,histo.GetNbinsX()+1):
  #   eff_tot.SetBinContent(bin,eff_bdt.GetBinContent(bin)*histo_presel_eff.GetBinContent(bin))
  #   eff_tot.SetBinError(bin,0)
  #   eff_bdt.SetBinError(bin,0)
  #   histo_presel_eff.SetBinError(bin,0)
  
<<<<<<< HEAD
  distribution.cd()
  eff_bdt.Write()
  eff_tot.Write()
  histo_presel_eff.Write()

  for histo,histo_eff in zip(histo_counts,histo_bdt):
    for bin in range(1,histo.GetNbinsX()+1):
      eff = histo_presel_eff.GetBinContent(bin)*histo_eff.GetBinContent(bin)
      bin_width = histo.GetBinWidth(bin)
      histo.SetBinContent(bin,histo.GetBinContent(bin)/eff/bin_width)
      histo.SetBinError(bin,histo.GetBinError(bin)/eff/bin_width)

    #this part can be skipped
    if not args.pt:
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

      tau.append(expo.GetParameter(1))
      err_tau.append(expo.GetParError(1))

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
      cv.Write()

    histo.Write()
  
  if not args.pt:

    histo_tau = TH1D("histo_tau",";eff;tau [ps]",len(tau),0,len(tau))
    for bin in range(1,len(tau)):
      histo_tau.SetBinContent(bin,tau[bin-1])
      histo_tau.SetBinError(bin,err_tau[bin-1])
      histo_tau.GetXaxis().SetBinLabel(bin,bin_label[bin-1])
    histo_tau.Write()

=======
  # distribution.cd()
  # eff_bdt.Write()
  # eff_tot.Write()
  # histo_presel_eff.Write()

  # for histo,histo_eff in zip(histo_counts,histo_bdt):
  #   for bin in range(1,histo.GetNbinsX()+1):
  #     eff = histo_presel_eff.GetBinContent(bin)*histo_eff.GetBinContent(bin)
  #     bin_width = histo.GetBinWidth(bin)
  #     histo.SetBinContent(bin,histo.GetBinContent(bin)/eff/bin_width)
  #     histo.SetBinError(bin,histo.GetBinError(bin)/eff/bin_width)

  #   #this part can be skipped
  #   if not args.pt:
  #     cv = TCanvas("ct_"+histo.GetName()[5:])
  #     cv.SetLogy()
  #     gStyle.SetOptStat(0)
  #     gStyle.SetOptFit(0)

  #     histo.UseCurrentStyle()
  #     histo.SetLineColor(1)
  #     histo.SetMarkerStyle(20)
  #     histo.SetMarkerColor(1)

  #     expo = TF1("","[0]*exp(-x/[1]/0.029979245800)",0,28)
  #     expo.SetParLimits(1,100,350)
  #     histo.Fit(expo,"MIR")

  #     pinfo2= TPaveText(0.5,0.5,0.91,0.9,"NDC")
  #     pinfo2.SetBorderSize(0)
  #     pinfo2.SetFillStyle(0)
  #     pinfo2.SetTextAlign(30+3)
  #     pinfo2.SetTextFont(42)
  #     string ='ALICE Internal, Pb-Pb 2018 {}-{}%'.format(0,90)
  #     pinfo2.AddText(string)
  #     string='#tau = {:.0f} #pm {:.0f} ps '.format(expo.GetParameter(1),expo.GetParError(1))
  #     pinfo2.AddText(string)  
  #     if expo.GetNDF()is not 0:
  #       string='#chi^{{2}} / NDF = {}'.format(expo.GetChisquare() / (expo.GetNDF()))
  #     pinfo2.AddText(string)  
  #     pinfo2.Draw()
  #     cv.Write()

  #   histo.Write()
>>>>>>> Easier logic for the distribution analysis macro
resultFile.Close()






