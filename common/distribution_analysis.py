#!/usr/bin/env python3

from ROOT import TF1, TH1D, TH2D, TCanvas, TFile, TPaveText, gDirectory, gStyle, gROOT, TIter, TKey, TAxis
import os
import yaml
import numpy as np
from array import array
import random
random.seed(1989)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to the YAML configuration file")
args = parser.parse_args()

gROOT.SetBatch()

expo = TF1("", "[0]*exp(-x/[1]/0.029979245800)", 0, 28)
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
  results = dict()
  hRawCounts = []
  for fix_eff in  params['BDT_EFFICIENCY']:
    if (float(fix_eff) > 0.68 or float(fix_eff) < 0.58):
      continue
    h2RawCounts = resultFile.Get('{}/RawCounts{}'.format(inDirName,fix_eff))
    h1RawCounts = h2RawCounts.ProjectionY()
    h1RawCounts.SetTitle(";#it{ct} (cm);dN/d#it{ct} (cm)^{-1}")
    h1RawCounts.Divide(h1PreselEff)
    h1RawCounts.Scale(1./float(fix_eff),'width')
    outDir.cd()
    h1RawCounts.UseCurrentStyle()
    h1RawCounts.Fit(expo,"MI")
    h1RawCounts.Write('ctSpectra{}'.format(fix_eff))
    hRawCounts.append(h1RawCounts)

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

  delta = TH1D("delta",";#tau (ps);Entries",30,200,300)
  for key, value in  results.items():
    delta.Fill(value[0])
    # if key == 0.6:
    #   continue
    # delta.Fill((value[0] - results[round(0.6,1)][0]) / results[round(0.6,1)][1])
  outDir.cd()
  delta.Write()

  syst = TH1D("syst",";#tau (ps);Entries",100,200,300)
  prob = TH1D("prob",";Lifetime fit probability;Entries",100,0,1)
  tmpCt = hRawCounts[0].Clone("tmpCt")
  
  combinations = set()
  for _ in range(10000):
    tmpCt.Reset()
    comboList=[]
    for iBin in range(1, tmpCt.GetNbinsX() + 1):
      index = random.randint(0,len(hRawCounts)-1)
      rdmCt = hRawCounts[index]
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

  syst.Write()
  prob.Write()
resultFile.Close()

