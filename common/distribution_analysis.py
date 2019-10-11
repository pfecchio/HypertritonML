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

expo = TF1("myexpo", "[0]*exp(-x/([1]*0.029979245800))/([1]*0.029979245800)", 0, 35)
expo.SetParLimits(1, 100, 350)

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

var = '#it{ct}'
unit = 'cm'

rangesFile = resultsSysDir + '/' + params['FILE_PREFIX'] + '_effranges.yaml'
with open(os.path.expandvars(rangesFile), 'r') as stream:
    try:
        ranges = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

file_name = resultsSysDir +  '/' + params['FILE_PREFIX'] + '_dist.root'
distribution = TFile(file_name,'recreate')

file_name = resultsSysDir +  '/' + params['FILE_PREFIX'] + '_results.root'
resultFile = TFile(file_name)

bkgModels = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']

for cclass in params['CENTRALITY_CLASS']:
  inDirName =  f"{cclass[0]}-{cclass[1]}"
  resultFile.cd(inDirName)
  outDir = distribution.mkdir(inDirName)
  cvDir = outDir.mkdir("canvas")

  h2PreselEff = resultFile.Get(f'{inDirName}/SelEff')
  h1PreselEff = h2PreselEff.ProjectionY()

  for i in range(1, h1PreselEff.GetNbinsX()+1):
    h1PreselEff.SetBinError(i, 0)

  h1PreselEff.SetTitle(f';{var} ({unit}); Preselection efficiency')
  h1PreselEff.UseCurrentStyle()
  h1PreselEff.SetMinimum(0)
  outDir.cd()
  h1PreselEff.Write("h1PreselEff")

  hRawCounts = []
  raws = []
  errs = []

  for model in bkgModels:
    h1RawCounts = h1PreselEff.Clone(f"best_{model}")
    h1RawCounts.Reset()

    for iBin in range(1, h1RawCounts.GetNbinsX()+1):
      h2RawCounts = resultFile.Get('{}/RawCounts{}_{}'.format(inDirName, ranges['BEST'][iBin-1], model))
      h1RawCounts.SetBinContent(iBin, h2RawCounts.GetBinContent(1, iBin) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCounts.GetBinWidth(iBin))
      h1RawCounts.SetBinError(iBin, h2RawCounts.GetBinError(1, iBin) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCounts.GetBinWidth(iBin))
      raws.append([])
      errs.append([])
      
      for eff in np.arange(ranges['SCAN'][iBin-1][0], ranges['SCAN'][iBin-1][1], ranges['SCAN'][iBin-1][2]):
        h2RawCounts = resultFile.Get(f'{inDirName}/RawCounts{eff:g}_{model}')
        raws[iBin-1].append(h2RawCounts.GetBinContent(1, iBin) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin))
        errs[iBin-1].append(h2RawCounts.GetBinError(1, iBin) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin))

    h1RawCounts.SetTitle(";#it{ct} (cm);dN/d#it{ct} (cm)^{-1}")
    outDir.cd()
    h1RawCounts.UseCurrentStyle()
    h1RawCounts.Fit(expo,"MI")
    h1RawCounts.Write()
    hRawCounts.append(h1RawCounts)

    cvDir.cd()
    myCv = TCanvas(f"ctSpectraCv_{model}")
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

  syst = TH1D("syst",";#tau (ps);Entries",300,100,400)
  prob = TH1D("prob",";Lifetime fit probability;Entries",100,0,1)
  pars = TH2D("pars",";#tau (ps);Normalisation;Entries",300,100,400,1000,4500,6500)
  tmpCt = hRawCounts[0].Clone("tmpCt")
  
  combinations = set()
  size = 100000

  for _ in range(size):
    tmpCt.Reset()
    comboList = []
    
    for iBin in range(1, tmpCt.GetNbinsX()+1):
      index = random.randint(0, len(raws[iBin-1])-1)
      comboList.append(index)
      tmpCt.SetBinContent(iBin, raws[iBin-1][index])
      tmpCt.SetBinError(iBin, errs[iBin-1][index])
      
    combo = (x for x in comboList)
    if combo in combinations:
      continue

    combinations.add(combo)
    tmpCt.Fit(expo,"MI")
    prob.Fill(expo.GetProb())
    if expo.GetChisquare() < 3 * expo.GetNDF():
      syst.Fill(expo.GetParameter(1))
      pars.Fill(expo.GetParameter(1),expo.GetParameter(0))

  syst.SetFillColor(600)
  syst.SetFillStyle(3345)
  syst.Write()
  prob.Write()
  pars.Write()

resultFile.Close()

