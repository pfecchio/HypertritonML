import uproot, numpy as np, pandas as pd 
from root_numpy import fill_hist, fill_graph 
from ROOT import TFile, TH2D,TCanvas, kBlack, kBlue, kRed, kOrange, TGraph, gROOT
from ROOT import TSpline3

gROOT.SetBatch()

df = uproot.open('../Tables/2Body/ct_tables/SignalTable_18_no_cuts.root')['SignalTable'].pandas.df() 

required_eff_list = [0.70,0.75,0.80,0.85,0.9,0.95]
tcanvas_list = []
graph_list=[]

cospa_bins = (2-np.abs(np.geomspace(-1,-2,400)))[::-1]
cospa_step_size = np.diff(cospa_bins)                                                                                     
ct_bins = np.geomspace(1,36,200)-1
ct_step_size = np.diff(ct_bins) 
hist_cosPA = TH2D("histo_cos_pa", '; c#it{t} (cm); # cosPA; Raw counts', len(ct_bins) - 1, ct_bins, len(cospa_bins)-1, cospa_bins) 
df['V0CosPA_abs'] = np.abs(df['V0CosPA'])
fill_hist(hist_cosPA, df[['ct','V0CosPA_abs']])
for required_eff in required_eff_list:
    cospa_bin_list = []
    for ct_bin in range(1, len(ct_bins)):
        gen = hist_cosPA.Integral(1,len(cospa_bins),ct_bin,ct_bin)
        count=0
        for cospa_bin in range(1, len(cospa_bins)):
            rec = hist_cosPA.Integral(cospa_bin, len(cospa_bins)-1, ct_bin, ct_bin)
            if (rec/gen<=required_eff and count==0):
                cospa_bin_list.append(cospa_bin-1)
                count = 1
            if count ==1:
                break
        if(rec/gen>=required_eff and count ==0):  #conservative estimation
            cospa_bin_list.append(cospa_bin)
    
    ct_array = ct_bins+ct_step_size/2
    cospa_array = cospa_bins[cospa_bin_list-1]+cospa_step_size[cospa_bin_list-1]/2
    fill = np.vstack((ct_array, cospa_array)).T
    graph_eff = TGraph(len(cospa_array))
    fill_graph(graph_eff,fill)
    graph_eff.SetName(f"{required_eff}")
    graph_eff.SetMarkerColor(kRed)
    graph_eff.SetMarkerStyle(8)
    graph_eff.SetMarkerSize(0.5)
    tcanvas_list.append(TCanvas(f"eff{required_eff}"))
    graph_list.append(graph_eff)

#write histo file
tfile = TFile('splines.root','recreate') 
for eff,graph,cv in zip(required_eff_list,graph_list, tcanvas_list):
    spl = TSpline3(f"spline_{eff}", graph)
    spl.SetLineColor(kOrange)
    spl.SetLineWidth(3)
    cv.cd()
    hist_cosPA.Draw("colz")
    graph.Draw("p")
    spl.Draw("same")
    cv.Write()
    cv.Close()
    spl.Write(f"spline_{eff}")
tfile.Close()

