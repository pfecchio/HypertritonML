#!/usr/bin/env python3

from ROOT import TF1, TH1D, TH2D, TCanvas, TFile, TPaveText, gDirectory, gStyle, gROOT, TIter, TKey, TAxis
import os
import yaml
import numpy as np
from array import array
import random
import math
random.seed(1989)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to the YAML configuration file")
args = parser.parse_args()

gROOT.SetBatch()

expo = TF1("myexpo", "[0]*exp(-x/[1]/0.029979245800)", 0, 28)
expo.SetParLimits(1, 100, 350)

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

var = '#it{ct}'
unit = 'cm'

file_name = resultsSysDir +  '/' + params['FILE_PREFIX'] + '_dist.root'
distribution = TFile(file_name,'recreate')

file_name = resultsSysDir +  '/' + params['FILE_PREFIX'] + '_results.root'
resultFile = TFile(file_name)

bkgModels = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']

for cclass in params['CENTRALITY_CLASS']:
  inDirName =  "{}-{}".format(cclass[0],cclass[1])
  resultFile.cd(inDirName)
  outDir = distribution.mkdir(inDirName)
  cvDir = outDir.mkdir("canvas")

  h2PreselEff = resultFile.Get("{}/SelEff".format(inDirName))
  h1PreselEff = h2PreselEff.ProjectionY()

  for i in range(1,h1PreselEff.GetNbinsX()+1):
    h1PreselEff.SetBinError(i,0)

  h1PreselEff.SetTitle(";{} ({}); Preselection efficiency".format(var,unit))
  h1PreselEff.UseCurrentStyle()
  h1PreselEff.SetMinimum(0)
  outDir.cd()
  h1PreselEff.Write("h1PreselEff")
  hRawCounts = []
  raws = [[],[],[],[],[],[],[],[]]
  trialNames = []
  for model in bkgModels:
    for fix_eff in  params['BDT_EFFICIENCY']:
      if (float(fix_eff) > 0.75 or float(fix_eff) < 0.6):
        continue
      h2RawCounts = resultFile.Get('{}/RawCounts{}_{}'.format(inDirName,fix_eff,model))
      h1RawCounts = h2RawCounts.ProjectionY()
      h1RawCounts.SetTitle(";#it{ct} (cm);dN/d#it{ct} (cm)^{-1}")
      h1RawCounts.Divide(h1PreselEff)
      h1RawCounts.Scale(1./float(fix_eff),'width')
      outDir.cd()
      for iBin in range(0,h1RawCounts.GetNbinsX()):
        raws[iBin].append(h1RawCounts.GetBinContent(iBin + 1))
      h1RawCounts.UseCurrentStyle()
      h1RawCounts.Fit(expo,"MI")
      h1RawCounts.Write('ctSpectra{}_{}'.format(fix_eff,model))
      hRawCounts.append(h1RawCounts)

      cvDir.cd()
      myCv = TCanvas("ctSpectraCv{}_{}".format(fix_eff,model))
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

  outDir.cd()

  syst = TH1D("syst",";#tau (ps);Entries",200,200,400)
  prob = TH1D("prob",";Lifetime fit probability;Entries",100,0,1)
  tmpCt = hRawCounts[0].Clone("tmpCt")
  
  combinations = set()
  flag = 0
  size = 100000
  for _ in range(100000):
    tmpCt.Reset()
    comboList=[]
    if random.uniform(0,1) < 100 / size:
      flag =  flag + 1
    for iBin in range(1, tmpCt.GetNbinsX() + 1):
      index = random.randint(0,len(hRawCounts)-1)
      rdmCt = hRawCounts[index]
      # if flag == 1:
      #   trialNames.append(hRawCounts[index].GetName())
      comboList.append(index)
      tmpCt.SetBinContent(iBin, rdmCt.GetBinContent(iBin))
      tmpCt.SetBinError(iBin, rdmCt.GetBinError(iBin))
    combo = (x for x in comboList)
    if combo in combinations:
      continue
    combinations.add(combo)
    tmpCt.Fit(expo,"MI")
    prob.Fill(expo.GetProb())
    if expo.GetChisquare() < 3 * expo.GetNDF():
      syst.Fill(expo.GetParameter(1))
    # if flag == 1:
    #   myCv = TCanvas("ctSpectraCv")
    #   pinfo2= TPaveText(0.5,0.6,0.91,0.9,"NDC")
    #   pinfo2.SetBorderSize(0)
    #   pinfo2.SetFillStyle(0)
    #   pinfo2.SetTextAlign(30+3)
    #   pinfo2.SetTextFont(42)
    #   string ='ALICE Internal, Pb-Pb 2018 {}-{}%'.format(0,90)
    #   pinfo2.AddText(string)
    #   string='#tau = {:.0f} #pm {:.0f} ps '.format(expo.GetParameter(1),expo.GetParError(1))
    #   pinfo2.AddText(string)  
    #   if expo.GetNDF()is not 0:
    #     string='#chi^{{2}} / NDF = {}'.format(expo.GetChisquare() / (expo.GetNDF()))
    #   pinfo2.AddText(string)
    #   tmpCt.Draw()
    #   tmpCt.SetMarkerStyle(20)
    #   tmpCt.SetMarkerColor(600)
    #   tmpCt.SetLineColor(600)
    #   pinfo2.Draw("x0")
    #   tmpSyst = tmpCt.Clone("hSyst")
    #   corSyst = tmpCt.Clone("hCorr")
    #   tmpSyst.SetFillStyle(0)
    #   corSyst.SetFillStyle(3345)
    #   for iBin in range(1, tmpCt.GetNbinsX() + 1):
    #     val = tmpSyst.GetBinContent(iBin)
    #     tmpSyst.SetBinError(iBin, np.std(raws[iBin - 1]))
    #     corSyst.SetBinError(iBin, 0.086 * val)
    #   tmpSyst.Draw("e2same")
    #   corSyst.Draw("e2same")
    #   outDir.cd()
    #   myCv.Write()
    #   flag = 2
  
  myCv = TCanvas("ctSpectraCv")
  pinfo2= TPaveText(0.5,0.6,0.91,0.9,"NDC")
  pinfo2.SetBorderSize(0)
  pinfo2.SetFillStyle(0)
  pinfo2.SetTextAlign(30+3)
  pinfo2.SetTextFont(42)
  string ='ALICE Internal, Pb-Pb 2018 {}-{}%'.format(0,90)
  pinfo2.AddText(string)
  expo7 = hRawCounts[7].GetFunction('myexpo')
  string='#tau = {:.0f} #pm {:.0f} ps '.format(expo7.GetParameter(1),expo7.GetParError(1))
  pinfo2.AddText(string)  
  if expo7.GetNDF()is not 0:
    string='#chi^{{2}} / NDF = {}'.format(expo7.GetChisquare() / (expo7.GetNDF()))
  pinfo2.AddText(string)
  hRawCounts[7].Draw()
  hRawCounts[7].SetMarkerStyle(20)
  hRawCounts[7].SetMarkerColor(600)
  hRawCounts[7].SetLineColor(600)
  pinfo2.Draw("x0")
  tmpSyst = hRawCounts[7].Clone("hSyst")
  corSyst = hRawCounts[7].Clone("hCorr")
  tmpSyst.SetFillStyle(0)
  corSyst.SetFillStyle(3345)
  for iBin in range(1, hRawCounts[7].GetNbinsX() + 1):
    val = tmpSyst.GetBinContent(iBin)
    tmpSyst.SetBinError(iBin, np.std(raws[iBin - 1]))
    corSyst.SetBinError(iBin, 0.086 * val)
  tmpSyst.Draw("e2same")
  corSyst.Draw("e2same")
  outDir.cd()
  myCv.Write()

  syst.SetFillColor(600)
  syst.SetFillStyle(3345)
  syst.Write()
  prob.Write()
  print(trialNames)
resultFile.Close()

