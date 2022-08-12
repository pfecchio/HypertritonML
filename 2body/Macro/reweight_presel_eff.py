import uproot
import numpy as np
import pandas as pd
import ROOT


def fill_hist(hist, dataframe, column):
    for val in dataframe[column]:
        hist.Fill(val)
    return hist

df_rec = uproot.open('../../Tables/2Body/SignalTable_upd_20g7.root')['SignalTable'].arrays(library='pd')
df_gen = uproot.open('../../Tables/2Body/SignalTable_upd_20g7.root')['GenTable'].arrays(library='pd')
cent_counts, cent_edges = uproot.open("/data/fmazzasc/PbPb_2body/2018/AnalysisResults_18qr.root")["AliAnalysisTaskHyperTriton2He3piML_custom_summary;1"][11].to_numpy()
df_rec  = df_rec.query('2<=pt<=10')

cent_bin_centers = (cent_edges[:-1]+cent_edges[1:])/2
cent_classes = [[0, 10], [10, 30], [30, 50], [50, 90]]
ct_bins = [1, 2, 4, 6, 8, 10, 14, 18, 23, 35]

n_events = []
eff_hists = []
for cent in cent_classes:
    cut = f'{cent[0]}<=centrality<={cent[1]}'
    df_rec_cent = df_rec.query(cut)
    df_gen_cent = df_gen.query(cut)
    cent_range_map = np.logical_and(cent_bin_centers > cent[0], cent_bin_centers < cent[1])
    counts_cent_range = cent_counts[cent_range_map]
    n_events.append(sum(counts_cent_range))
    reco_hist = ROOT.TH1F(f'eff_ct_cent_{cent[0]}_{cent[1]}', f'eff_ct_cent_{cent[0]}_{cent[1]}', len(ct_bins)-1, np.array(ct_bins, dtype=np.float32))
    gen_hist = ROOT.TH1F(f'gen_ct_cent_{cent[0]}_{cent[1]}', f'gen_ct_cent_{cent[0]}_{cent[1]}', len(ct_bins)-1, np.array(ct_bins, dtype=np.float32))
    fill_hist(reco_hist, df_rec_cent, 'ct')
    fill_hist(gen_hist, df_gen_cent, 'ct')
    reco_hist.Sumw2()
    reco_hist.Divide(gen_hist)
    eff_hists.append(reco_hist)

ct_bins = [1, 2, 4, 6, 8, 10, 14, 18, 23, 35]

eff_ct_090 = ROOT.TH1F("eff_ct_090", "eff_ct", len(ct_bins)-1, np.array(ct_bins,dtype=np.float32))
gen_ct_090 = ROOT.TH1F("gen_ct_090", "gen_ct", len(ct_bins)-1, np.array(ct_bins,dtype=np.float32))

fill_hist(eff_ct_090, df_rec, 'ct')
fill_hist(gen_ct_090, df_gen, 'ct')
eff_ct_090.Sumw2()
eff_ct_090.Divide(gen_ct_090)

eff_ct_090_rew = ROOT.TH1F("eff_ct_090_rew", "eff_ct", len(ct_bins)-1, np.array(ct_bins,dtype=np.float32))

print(n_events)
print(sum(n_events))

for i in range(len(ct_bins)-1):
    bin_content = 0
    bin_error = 0
    for n_ev,hist_cent in zip(n_events, eff_hists):
        bin_content += n_ev*hist_cent.GetBinContent(i+1)
        bin_error += n_ev*hist_cent.GetBinError(i+1)
        print("cent bin error: ", bin_error)
    
    bin_content /= sum(n_events)
    bin_error /= sum(n_events)
    print(bin_error)
    eff_ct_090_rew.SetBinContent(i+1, bin_content)
    eff_ct_090_rew.SetBinError(i+1, bin_error)

out_file = ROOT.TFile("eff_ct_090_rew.root", "RECREATE")
eff_ct_090.Write()
eff_ct_090_rew.Write()

eff_ct_090.Divide(eff_ct_090_rew)

cv = ROOT.TCanvas("cv", "cv", 800, 600)
eff_ct_090.Draw()
eff_ct_090.GetXaxis().SetTitle("#it{c}_{t} (cm)")
eff_ct_090.GetYaxis().SetTitle("Eff_{old} / Eff_{rew}")
cv.Write()
out_file.Close()