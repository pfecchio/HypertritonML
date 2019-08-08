import pyroot_plot as prp
from ROOT import (TH1D, AliPID, TCanvas, TFile, TGaxis, TLegend,
                  TLorentzVector, TPad, TTree, gROOT)

TGaxis.SetMaxDigits(4)

color_a = [prp.kTempsScale6, prp.kTempsScale5, prp.kTempsScale4]
color_m = [prp.kTempsScale0, prp.kTempsScale1, prp.kTempsScale2]

label_centrality = ['0-10', '10-50', '50-90']
label_am = ['antihyper', 'hyper']

label_array = [x + '_' + y for x in label_am for y in label_centrality]

dict_hist_eff = {}

# extract histos from file
input_file = TFile('~/3body_workspace/results/eff_hist.root', 'read')

for lab in label_array:
    hist = input_file.Get('fHistEfficiency_{}'.format(lab))
    hist.SetDirectory(0)

    dict_hist_eff[lab] = hist

input_file.Close()

# output file for the beautified plots
output_file = TFile('~/3body_workspace/results/eff_plot.root', 'recreate')

#-----------------------------------------------------------------------------#
# efficiency for antimatter in different centrality classes
#-----------------------------------------------------------------------------#
c = TCanvas('c_eff_cent_a', '', 700, 500)

legend = TLegend(0.165, 0.650, 0.474, 0.865, 'C')
legend.SetFillStyle(0)
legend.SetTextSize(18)
legend.SetHeader('anti-hypertriton')

for cent in range(3):
    h = dict_hist_eff[label_am[0] + '_' + label_centrality[cent]]
    h.SetDirectory(0)

    prp.histo_makeup(h, color=color_a[cent], x_title='#it{p}_{T} (GeV/#it{c} )',
                     y_title='Efficiency #times Acceptance')

    legend.AddEntry(h, label_centrality[cent])

    dict_hist_eff[label_am[0] + '_' + label_centrality[cent]] = h

    if cent == 0:
        h.Draw('')
    else:
        h.Draw('same')

legend.Draw()
c.Write()

#-----------------------------------------------------------------------------#
# efficiency for matter in different centrality classes
#-----------------------------------------------------------------------------#
c = TCanvas('c_eff_cent_m', '', 700, 500)

legend = TLegend(0.165, 0.650, 0.474, 0.865, 'C')
legend.SetFillStyle(0)
legend.SetTextSize(18)
legend.SetHeader('hypertriton')

for cent in range(3):
    h = dict_hist_eff[label_am[1] + '_' + label_centrality[cent]]
    h.SetDirectory(0)

    prp.histo_makeup(h, color=color_m[cent])

    legend.AddEntry(h, label_centrality[cent])

    dict_hist_eff[label_am[1] + '_' + label_centrality[cent]] = h

    if cent == 0:
        h.Draw('')
    else:
        h.Draw('same')

legend.Draw()
c.Write()

#-----------------------------------------------------------------------------#
# matter-antimatter comparison
#-----------------------------------------------------------------------------#
c = TCanvas('c_eff_comparison', '', 1600, 550)
c.Divide(3, 1)

h3 = []
legend = []

for cent in range(1, 4):
    c.cd(cent)

    h_a = dict_hist_eff[label_am[0] + '_' + label_centrality[cent - 1]]
    h_a.SetDirectory(0)

    h_a.GetYaxis().SetRangeUser(-0.12, 0.36)

    h_m = dict_hist_eff[label_am[1] + '_' + label_centrality[cent - 1]]
    h_m.SetDirectory(0)

    h_m.GetYaxis().SetRangeUser(-0.12, 0.36)

    pad1 = TPad('pad1', 'pad1', 0, 0.3, 1, 1.0)
    pad1.SetBottomMargin(0)  # Upper and lower plot are joined
    pad1.Draw()             # Draw the upper pad: pad1
    pad1.cd(1)               # pad1 becomes the current pad

    h_a.SetStats(0)
    h_a.Draw()

    h2opt = h_m.GetOption() + 'same'
    h_m.Draw(h2opt)

    legend.append(TLegend(0.165, 0.650, 0.474, 0.865, 'C'))
    legend[cent-1].SetFillStyle(0)
    legend[cent-1].SetTextSize(18)

    legend[cent-1].SetHeader('cent. class {}'.format(label_centrality[cent - 1]))
    legend[cent-1].AddEntry(h_a, 'anti-hypertriton')
    legend[cent-1].AddEntry(h_m, 'hypertriton')

    legend[cent-1].Draw()

    # axis = TGaxis(-5, 20, -5, 220, 20, 220, 510, '')
    axis = h_a.GetYaxis()
    axis.SetDecimals()
    axis.SetLabelFont(43)  # Absolute font size in pixel (precision 3)
    axis.SetLabelSize(18)
    axis.SetTitleFont(43)
    axis.SetTitleSize(20)
    axis.SetTitleOffset(1.6)
    axis.Draw()
    c.Update()

    # lower plot will be in pad
    c.cd(cent)          # Go back to the main canvas before defining pad2
    pad2 = TPad('pad2', 'pad2', 0, 0., 1., 0.30)
    pad2.SetTopMargin(0.0)
    pad2.SetBottomMargin(0.3)
    pad2.SetGridy(1)
    pad2.Draw()
    pad2.cd()       # pad2 becomes the current pad

    # Define the ratio plot
    h3.append(h_a.Clone('h3'))
    h3[cent-1].SetDirectory(0)
    h3[cent-1].SetLineColor(prp.kPurpleC)
    h3[cent-1].SetMarkerColor(prp.kPurpleC)
    h3[cent-1].SetMarkerSize(0.5)
    h3[cent-1].Sumw2()
    h3[cent-1].SetStats(0)  # No statistics on lower plot

    h3[cent-1].Add(h_m, -1.)

    # Remove the ratio title
    h3[cent-1].SetTitle('')
    h3[cent-1].GetYaxis().SetTitle('diff')
    h3[cent-1].GetYaxis().SetNdivisions(505)
    h3[cent-1].GetYaxis().SetTitleSize(20)
    h3[cent-1].GetYaxis().SetTitleFont(43)
    h3[cent-1].GetYaxis().SetTitleOffset(1.6)
    h3[cent-1].GetYaxis().SetLabelFont(43)
    h3[cent-1].GetYaxis().SetLabelSize(16)
    h3[cent-1].GetYaxis().SetRangeUser(-0.022, 0.012)

    # X axis ratio plot settings
    h3[cent-1].GetXaxis().SetTitle('#it{p}_{T} (GeV/#it{c} )')
    h3[cent-1].GetXaxis().SetTitleSize(20)
    h3[cent-1].GetXaxis().SetTitleFont(43)
    h3[cent-1].GetXaxis().SetTitleOffset(3.5)
    h3[cent-1].GetXaxis().SetLabelFont(43)
    h3[cent-1].GetXaxis().SetLabelOffset(0.02)
    # Absolute font size in pixel(precision 3)
    h3[cent-1].GetXaxis().SetLabelSize(18)

    h3[cent-1].SetMarkerStyle(21)
    h3[cent-1].Draw('ep')  # Draw the ratio plot
    c.Update()

c.Write()
c.SaveAs('comparison.pdf')
c.Close()

output_file.Write()
output_file.Close()
