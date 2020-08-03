import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ROOT
import uproot

ROOT.gROOT.ForceStyle()
tfile_res = ROOT.TFile("ct_analysis_o2_data_results.root") 
tfile_em = ROOT.TFile("ct_analysis_o2_pion_LS_results.root") 

dirs = uproot.open("ct_analysis_o2_data_results.root")["0-90"].keys()[:-1]
keys = []
for i in range(len(dirs)):
    dirs[i] = dirs[i].decode('utf-8')[:-2]
    keys.append(uproot.open("ct_analysis_o2_data_results.root")["0-90"][dirs[i]].keys()[0][:-2].decode("utf-8"))


hist_data_list = []
hist_em_list = []
for i in range(len(dirs)):
    hist_data_list.append(tfile_res.Get(f"0-90/{dirs[i]}/{keys[i]}"))
    hist_em_list.append(tfile_em.Get(f"0-90/{dirs[i]}/{keys[i]}"))

out_file = ROOT.TFile("data_LS_comp.root", "recreate")
for key, histo_data, histo_em in zip(keys, hist_data_list, hist_em_list):
    histo_em.Scale((histo_data.Integral(22, 40)+histo_data.Integral(1, 10))/(histo_em.Integral(22, 40)+histo_em.Integral(1, 10)))
    for iBin in range(1, histo_data.GetNbinsX() + 1):
        histo_em.SetBinError(iBin, np.sqrt(histo_em.GetBinContent(iBin)))
    histo_data.SetTitle("; m (d + p + #pi) (GeV/#it{c}^{2}); Counts")
    histo_data.SetMinimum(0)
    histo_em.SetTitle("Like Sign")   
    canv = ROOT.TCanvas(key)
    canv.Divide(1,2,0,0)

    canv.cd(1)
    
    canv.GetPad(1).SetRightMargin(.01)
    histo_data.Draw()
    histo_data.SetMinimum(-5)
    histo_data.UseCurrentStyle()
    histo_data.SetStats(0)
    histo_em.SetMarkerColor(ROOT.kRed)
    histo_em.SetLineColor(ROOT.kRed)    
    histo_em.Draw("same")
    leg = ROOT.TLegend(0.7,0.6,0.9,0.8)
    leg.AddEntry(histo_data,"Data")
    leg.AddEntry(histo_em, "Like Sign")
    leg.Draw()
    histo_data.GetYaxis().SetTitleOffset(1.3)
    histo_data.GetYaxis().SetTitleSize(28)
    


    canv.cd(2)
    canv.GetPad(2).SetRightMargin(.01)
    canv.GetPad(2).SetBottomMargin(0.2)
    histo_sum = histo_data.Clone("Sum")
    histo_sum.Add(histo_em, -1.)
    histo_sum.UseCurrentStyle()
    histo_sum.SetStats(0)
    histo_sum.SetMarkerColor(ROOT.kMagenta)
    histo_sum.SetLineColor(ROOT.kMagenta)    
    histo_sum.Draw()
    leg2 = ROOT.TLegend(0.7,0.6,0.9,0.97)
    leg2.AddEntry(histo_sum,"Data - Like Sign")
    leg2.Draw()
    histo_sum.GetXaxis().SetTitleOffset(2.5)
    histo_sum.GetYaxis().SetTitleOffset(1.3)
    histo_sum.GetYaxis().SetTitleSize(28)
    histo_sum.GetXaxis().SetTitleSize(28)


    canv.Write()

