#!/usr/bin/env python3

import os
import uproot
from ROOT import TF1, TH1D, TH2D, TCanvas, TFile, TPaveText, gDirectory, gStyle, gROOT, TIter, TKey, TClass

gROOT.SetBatch()

pT_bins = [[2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 9]]

eff_010 = [['pT23', 0.51], ['pT34', 0.71], ['pT45', 0.67], ['pT56', 0.72], ['pT67', 0.84], ['pT79', 0.82]]
eff_3050 = [['pT23', 0.72], ['pT34', 0.81], ['pT45', 0.88], ['pT56', 0.91], ['pT67', 0.97], ['pT79', 0.99]]
eff_3050_bis = [['pT23', 0.80], ['pT34', 0.80], ['pT45', 0.80], ['pT56', 0.91], ['pT67', 0.95], ['pT79', 0.99]]

var = 'p_{T}'
unit = 'GeV/c'

results_dir = os.environ['HYPERML_RESULTS_2']

out_file_path = results_dir + '/' + 'pT_dist.root'
output_file = TFile(out_file_path, 'recreate')

file_path = '~/Desktop/results/pt_analysis_results_2mev.root'
input_file_010 = TFile(file_path, 'read')

file_path = '~/Desktop/results/pt_analysis_results_3.6mev.root'
input_file_3050 = TFile(file_path, 'read')

data_path = os.path.expandvars('$HYPERML_TABLES_2/DataTable.root')
hist_centrality = uproot.open(data_path)['EventCounter']

n_events_010 = sum(hist_centrality[1:10])
n_events_3050 = sum(hist_centrality[30:50])

bw_file = TFile(os.environ['HYPERML_UTILS'] + '/BlastWaveFits.root')
bw = bw_file.Get('BlastWave/BlastWave0')


#################################################
# spectrum in 0-10% centrality class
#################################################

input_file_010.cd('0-10')

# create histograms for preselection and BDT efficiencies
histo_presel_eff_010 = gROOT.FindObject('SelEff').ProjectionX()
histo_presel_eff_010.SetTitle(';{} [{}]; preselection efficiency '.format(var, unit))

histo_BDTeff_010 = histo_presel_eff_010.Clone()
histo_BDTeff_010.SetName('BDTeff')
histo_BDTeff_010.SetTitle('BDTEff;{} [{}]; BDT efficiency '.format(var, unit))

# create histo for the total efficiency
eff_tot_010 = histo_presel_eff_010.Clone()
eff_tot_010.SetName('eff_tot')
eff_tot_010.SetTitle(';{} [{}]; total efficiency'.format(var, unit))

# create histogram for the production yield
histo_production_010 = gROOT.FindObject('RawCounts').ProjectionX()
histo_production_010.SetName('pT_spectrum')
histo_production_010.SetTitle(';p_{T} GeV/c;1/ (N_{ev}) d^{2}N/(dy dp_{T}) x B.R. (GeV/c)^{-1}')

for bin_index in range(1, 7):
    index = bin_index - 1

    # compute binwidth and total efficiency
    bin_widht = pT_bins[index][1] - pT_bins[index][0]
    eff = histo_presel_eff_010.GetBinContent(bin_index)
    bdt_eff = eff_010[index][1]
    tot_eff = eff * bdt_eff 

    # get raw counts from the correct histo
    raw_histo = gROOT.FindObject('RawCounts{}'.format(eff_010[index][1])).ProjectionX()
    raw_counts = raw_histo.GetBinContent(bin_index)
    raw_counts_error = raw_histo.GetBinError(bin_index)

    # fill production histo
    histo_production_010.SetBinContent(bin_index, raw_counts / tot_eff / bin_widht / n_events_010 / 0.25)
    histo_production_010.SetBinError(bin_index, raw_counts_error / tot_eff / bin_widht / n_events_010 / 0.25)

    # fill BDTeff and total eff histos
    eff_tot_010.SetBinContent(bin_index, tot_eff)
    histo_BDTeff_010.SetBinContent(bin_index, bdt_eff)

dir_010 = output_file.mkdir('0-10')
dir_010.cd()

histo_production_010.Write()
eff_tot_010.Write()
histo_BDTeff_010.Write()

#################################################
# spectrum in 30-50% centrality class
#################################################

input_file_3050.cd('30-50')

# create histograms for preselection and BDT efficiencies
histo_presel_eff_3050 = gROOT.FindObject('SelEff').ProjectionX()
histo_presel_eff_3050.SetTitle(';{} [{}]; preselection efficiency '.format(var, unit))

histo_BDTeff_3050 = histo_presel_eff_3050.Clone()
histo_BDTeff_3050.SetName('BDTeff')
histo_BDTeff_3050.SetTitle('BDTEff;{} [{}]; BDT efficiency '.format(var, unit))

# create histo for the total efficiency
eff_tot_3050 = histo_presel_eff_3050.Clone()
eff_tot_3050.SetName('eff_tot')
eff_tot_3050.SetTitle(';{} [{}]; total efficiency'.format(var, unit))

# create histogram for the production yield
histo_production_3050 = gROOT.FindObject('RawCounts').ProjectionX()
histo_production_3050.SetName('pT_spectrum')
histo_production_3050.SetTitle(';p_{T} GeV/c;1/ (N_{ev}) d^{2}N/(dy dp_{T}) x B.R. (GeV/c)^{-1}')

for bin_index in range(1, 7):
    index = bin_index - 1

    # compute binwidth and total efficiency
    bin_widht = pT_bins[index][1] - pT_bins[index][0]
    eff = histo_presel_eff_3050.GetBinContent(bin_index)
    bdt_eff = eff_3050[index][1]
    tot_eff = eff * bdt_eff 

    # get raw counts from the correct histo
    raw_histo = gROOT.FindObject('RawCounts{}'.format(eff_3050[index][1])).ProjectionX()
    raw_counts = raw_histo.GetBinContent(bin_index)
    raw_counts_error = raw_histo.GetBinError(bin_index)

    # fill production histo
    histo_production_3050.SetBinContent(bin_index, raw_counts / tot_eff / bin_widht / n_events_3050 / 0.25)
    histo_production_3050.SetBinError(bin_index, raw_counts_error / tot_eff / bin_widht / n_events_3050 / 0.25)

    # fill BDTeff and total eff histos
    eff_tot_3050.SetBinContent(bin_index, tot_eff)
    histo_BDTeff_3050.SetBinContent(bin_index, bdt_eff)

dir_3050 = output_file.mkdir('30-50')
dir_3050.cd()

histo_production_3050.Write()
eff_tot_3050.Write()
histo_BDTeff_3050.Write()

#################################################
# spectrum in 30-50% centrality class bis
#################################################

input_file_3050.cd('30-50')

# create histograms for preselection and BDT efficiencies
histo_presel_eff_3050_bis = gROOT.FindObject('SelEff').ProjectionX()
histo_presel_eff_3050_bis.SetTitle(';{} [{}]; preselection efficiency '.format(var, unit))

histo_BDTeff_3050_bis = histo_presel_eff_3050_bis.Clone()
histo_BDTeff_3050_bis.SetName('BDTeff')
histo_BDTeff_3050_bis.SetTitle('BDTEff;{} [{}]; BDT efficiency '.format(var, unit))

# create histo for the total efficiency
eff_tot_3050_bis = histo_presel_eff_3050_bis.Clone()
eff_tot_3050_bis.SetName('eff_tot')
eff_tot_3050_bis.SetTitle(';{} [{}]; total efficiency'.format(var, unit))

# create histogram for the production yield
histo_production_3050_bis = gROOT.FindObject('RawCounts').ProjectionX()
histo_production_3050_bis.SetName('pT_spectrum')
histo_production_3050_bis.SetTitle(';p_{T} GeV/c;1/ (N_{ev}) d^{2}N/(dy dp_{T}) x B.R. (GeV/c)^{-1}')

for bin_index in range(1, 7):
    index = bin_index - 1

    # compute binwidth and total efficiency
    bin_widht = pT_bins[index][1] - pT_bins[index][0]
    eff = histo_presel_eff_3050_bis.GetBinContent(bin_index)
    bdt_eff = eff_3050_bis[index][1]
    tot_eff = eff * bdt_eff 

    # get raw counts from the correct histo
    raw_histo = gROOT.FindObject('RawCounts{}'.format(eff_3050_bis[index][1])).ProjectionX()
    raw_counts = raw_histo.GetBinContent(bin_index)
    raw_counts_error = raw_histo.GetBinError(bin_index)

    # fill production histo
    histo_production_3050_bis.SetBinContent(bin_index, raw_counts / tot_eff / bin_widht / n_events_3050 / 0.25)
    histo_production_3050_bis.SetBinError(bin_index, raw_counts_error / tot_eff / bin_widht / n_events_3050 / 0.25)

    # fill BDTeff and total eff histos
    eff_tot_3050_bis.SetBinContent(bin_index, tot_eff)
    histo_BDTeff_3050_bis.SetBinContent(bin_index, bdt_eff)

dir_3050_bis = output_file.mkdir('30-50bis')
dir_3050_bis.cd()

histo_production_3050_bis.Write()
eff_tot_3050_bis.Write()
histo_BDTeff_3050_bis.Write()

canvas = TCanvas('c010', '', 800, 600)

histo_production_010.Divide(bw)
histo_production_010.Draw()

output_file.cd()
canvas.Write()

output_file.Close()

input_file_010.Close()
input_file_3050.Close()