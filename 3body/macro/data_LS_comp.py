import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ROOT
import uproot
from root_numpy import fill_hist
import os 


data_file = os.environ['HYPERML_RESULTS_3'] + "/ct_analysis_o2_results.root"
bkg_file = os.environ['HYPERML_RESULTS_3'] + "/ct_analysis_o2_LS_results.root"
eff_file = os.environ['HYPERML_EFFICIENCIES_3'] + "/PreselEff_cent090.root"
out_file = os.environ['HYPERML_RESULTS_3'] + "data_LS_comparison.root"

side_bands_norm = True


tfile_res = ROOT.TFile(data_file) 
tfile_bkg = ROOT.TFile(bkg_file) 


ct_bins = np.array(uproot.open(eff_file)["PreselEff"].edges[1],"double")

dirs = uproot.open(data_file)["0-90"].keys()[:-1]
keys = []
bdt_eff = []
for i in range(len(dirs)):
    dirs[i] = dirs[i].decode('utf-8')[:-2]
    keys.append(uproot.open(data_file)["0-90"][dirs[i]].keys()[0][:-2].decode("utf-8"))
    keys[0] = "ct24_pT210_cen090_eff0.60"
    bdt_eff.append(float(keys[i][-4:]))


hist_data_list = []
hist_bkg_list = []
counts_list = []

histo_subtr_list = []
for i in range(len(dirs)):
    hist_data_list.append(tfile_res.Get(f"0-90/{dirs[i]}/{keys[i]}"))
    hist_bkg_list.append(tfile_bkg.Get(f"0-90/{dirs[i]}/{keys[i]}"))

out_file = ROOT.TFile(out_file, "recreate")
for key, histo_data, histo_bkg in zip(keys, hist_data_list, hist_bkg_list):

    if(side_bands_norm):
        histo_bkg.Scale((histo_data.Integral(22, 40)+histo_data.Integral(1, 10))/(histo_bkg.Integral(22, 40)+histo_bkg.Integral(1, 10)))

    for iBin in range(1, histo_data.GetNbinsX() + 1):
        histo_bkg.SetBinError(iBin, np.sqrt(histo_bkg.GetBinContent(iBin)))
    histo_data.SetTitle("; m (d + p + #pi) (GeV/#it{c}^{2}); Counts")
    histo_data.SetMinimum(0)
    histo_bkg.SetTitle("Like Sign")   
    canv = ROOT.TCanvas(key)
    canv.Divide(1,2,0,0)

    canv.cd(1)
    
    canv.GetPad(1).SetRightMargin(.01)
    histo_data.Draw()
    histo_data.SetMinimum(-5)
    histo_data.UseCurrentStyle()
    histo_data.SetStats(0)
    histo_bkg.SetMarkerColor(ROOT.kRed)
    histo_bkg.SetLineColor(ROOT.kRed)    
    histo_bkg.Draw("same")
    leg = ROOT.TLegend(0.5,0.6,0.9,0.9)
    leg.AddEntry(histo_data,"Data ")
    leg.AddEntry(histo_bkg, "Like Sign")
    leg.Draw()
    histo_data.GetYaxis().SetTitleOffset(1.3)
    histo_data.GetYaxis().SetTitleSize(28)
    canv.cd(2)
    canv.GetPad(2).SetRightMargin(.01)
    canv.GetPad(2).SetBottomMargin(0.2)
    histo_sum = histo_data.Clone("Sum")
    histo_sum.Add(histo_bkg, -1.)

    fit_tpl = ROOT.TF1('fitTpl', 'pol0(0)+gausn(1)', 0, 5)

    # redefine parameter names for the bkg_model
    fit_tpl.SetParName(0, 'B_0')

    # define parameter names for the signal fit
    fit_tpl.SetParName(1, 'N_{sig}')
    fit_tpl.SetParName(2, '#mu')
    fit_tpl.SetParName(3, '#sigma')
    # define parameter values and limits
    fit_tpl.SetParameter(1, 40)
    fit_tpl.SetParLimits(1, 0, 1000)
    fit_tpl.SetParameter(2, 2.991)
    fit_tpl.SetParLimits(2, 2.990, 2.993)
    fit_tpl.SetParLimits(3, 0.001, 0.005)   
    fit_tpl.SetNpx(300) 


    histo_sum.Fit("fitTpl", "QRL", "")
    counts_list.append(histo_sum.GetFunction("fitTpl").GetParameter(1)/histo_sum.GetBinWidth(1))
    histo_sum.UseCurrentStyle()
    histo_sum.SetMarkerColor(ROOT.kMagenta)
    histo_sum.SetLineColor(ROOT.kMagenta)    
    histo_sum.Draw()
    leg2 = ROOT.TLegend(0.6,0.6,0.9,0.97)
    leg2.AddEntry(histo_sum,"Data - Like Sign")
    leg2.Draw()
    histo_sum.GetXaxis().SetTitleOffset(2.5)
    histo_sum.GetYaxis().SetTitleOffset(1.3)
    histo_sum.GetYaxis().SetTitleSize(28)
    histo_sum.GetXaxis().SetTitleSize(28)
    canv.Write()
    histo_subtr_list.append(histo_sum)


presel_eff_array = uproot.open(eff_file)["PreselEff"].values[0]
counts_array = np.array(counts_list)
bdt_eff_array = np.array(bdt_eff)
ct_spectrum_array = counts_array/bdt_eff_array/presel_eff_array
ct_spectrum_array_error = np.sqrt(np.abs(counts_array))/bdt_eff_array/presel_eff_array


bin_width = ct_bins[1:]-ct_bins[:-1]
ct_spectrum = ROOT.TH1D("ct spectrum", ";#it{c}t (cm);d#it{N}/d(#it{c}t) [(cm)^{-1}]", len(ct_bins)-1, ct_bins)

for iBin in range(1, ct_spectrum.GetNbinsX() + 1):
    ct_spectrum.SetBinContent(iBin, ct_spectrum_array[iBin-1]/bin_width[iBin-1])
    ct_spectrum.SetBinError(iBin, ct_spectrum_array_error[iBin-1]/bin_width[iBin-1])



### Lifetime fit
expo = ROOT.TF1("myexpo", "[0]*exp(-x/([1]*0.029979245800))/([1]*0.029979245800)", ct_bins[0], ct_bins[-1])
expo.SetParLimits(1, 100, 350)
ct_spectrum.Fit(expo, "MI0+", "", 2, 30)
canv = ROOT.TCanvas("ct_spectrum")
canv.SetLogy()
ct_spectrum.Draw()
fit_function = ct_spectrum.GetFunction("myexpo")
fit_function.SetLineColor(ROOT.kRed)
fit_function.Draw("same")
canv.Write()

out_file.Close()