import uproot, numpy as np, pandas as pd 
from root_numpy import fill_hist, fill_graph 
from ROOT import TFile, TH2D,TCanvas, kBlack, kBlue, kRed, TGraph, gROOT
gROOT.SetBatch()

df = uproot.open('../Tables/2Body/ct_tables/SignalTable_18_no_cuts.root')['SignalTable'].pandas.df() 
inf = 0
sup = 1
steps = 101
required_eff_list = [0.70,0.75,0.80,0.85,0.9,0.95]
tcanvas_list = []
graph_list=[]

cospa_bins,step_size = np.linspace(inf, sup , steps, retstep=True)
ct_bins = np.arange(0,35,0.02) 
hist_cosPA = TH2D("histo_cos_pa", ';# cosPA; c#it{t} (cm);Raw counts', len(cospa_bins)-1, cospa_bins, len(ct_bins) - 1, ct_bins) 
df['V0CosPA'] = np.abs(df['V0CosPA'])
fill_hist(hist_cosPA, df[['V0CosPA', 'ct']])

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
        if(rec/gen>=required_eff and count ==0):
            cospa_bin_list.append(cospa_bin)
    
    ct_array = (ct_bins+0.01)[:-1]
    cospa_array = np.array(cospa_bin_list)*step_size-(len(cospa_bins)-1)*step_size+sup
    print(cospa_array)
    fill = np.vstack((cospa_array, ct_array)).T
    graph_eff = TGraph(len(cospa_array))
    fill_graph(graph_eff,fill)
    graph_eff.SetName(f"{required_eff}")
    graph_eff.SetMarkerColor(kRed)
    graph_eff.SetMarkerStyle(8)
    graph_eff.SetMarkerSize(0.5)
    tcanvas_list.append(TCanvas(f"eff{required_eff}"))
    graph_list.append(graph_eff)

tfile = TFile('eff_cospa_ct_new_.root','recreate') 
for graph,cv in zip(graph_list, tcanvas_list):
    cv.cd()
    hist_cosPA.Draw("colz")
    graph.Draw("p")
    cv.Write()
    cv.Close()
tfile.Close()

        
