import math

import ctime
import pyroot_plot as prp
from ROOT import (TH1D, AliPID, TCanvas, TFile, TGaxis, TLegend,
                  TLorentzVector, TPad, TTree, gROOT)


def pot2(x):
    return x*x


def hypot4(x0, x1, x2, x3):
    return math.sqrt(pot2(x0) + pot2(x1) + pot2(x2) + pot2(x3))


n_bins = 20
test_mode = True
control = 0

TGaxis.SetMaxDigits(4)

label_centrality = ['0-10', '10-50', '50-90']
label_am = ['antihyper', 'hyper']


input_file = TFile('~/data/3body_hypetriton_data/train_tree/mc/HyperTritonTree.root', 'read')
# input_file = TFile('~/data/3body_hypetriton_data/train_tree/mc/HyperTritonTreeTest.root', 'read')

tree = input_file.fHypertritonTree

hist_sim = {}
hist_rec = {}

label_array = []

for lab in label_centrality:
    for am in label_am:
        label = '{}_{}'.format(am, lab)

        hist_sim[label] = TH1D('fHistSim_{}'.format(label), '', n_bins, 0, 10)
        hist_rec[label] = TH1D('fHistRec_{}'.format(label), '', n_bins, 0, 10)

        hist_sim[label].SetDirectory(0)
        hist_rec[label].SetDirectory(0)

        label_array.append(label)

# main loop over the events
for ev in tree:
    centrality = ev.REvent.fCent

    c_lab = ''

    if centrality <= 10.:
        c_lab = '{}'.format(label_centrality[0])

    elif centrality <= 50.:
        c_lab = '{}'.format(label_centrality[1])

    elif centrality <= 90.:
        c_lab = '{}'.format(label_centrality[2])

    if centrality > 90.:
        continue

    # loop over the simulated hypertritons
    for sim in ev.SHypertriton:
        charge = sim.fPdgCode > 0

        hyp = TLorentzVector()
        deu = TLorentzVector()
        p = TLorentzVector()
        pi = TLorentzVector()

        e_deu = hypot4(sim.fPxDeu, sim.fPyDeu, sim.fPzDeu, AliPID.ParticleMass(AliPID.kDeuteron))
        e_p = hypot4(sim.fPxP, sim.fPyP, sim.fPzP, AliPID.ParticleMass(AliPID.kProton))
        e_pi = hypot4(sim.fPxPi, sim.fPyPi, sim.fPzPi, AliPID.ParticleMass(AliPID.kPion))

        deu.SetPxPyPzE(sim.fPxDeu, sim.fPyDeu, sim.fPzDeu, e_deu)
        p.SetPxPyPzE(sim.fPxP, sim.fPyP, sim.fPzP, e_p)
        pi.SetPxPyPzE(sim.fPxPi, sim.fPyPi, sim.fPzPi, e_pi)

        hyp = deu + p + pi

        label = '{}_'.format(label_am[charge]) + c_lab
        hist_sim[label].Fill(hyp.Pt())

    # loop over the reconstructed hypertritons
    for rec in ev.RHypertriton:
        hyp = TLorentzVector()
        deu = TLorentzVector()
        p = TLorentzVector()
        pi = TLorentzVector()

        e_deu = hypot4(rec.fPxDeu, rec.fPyDeu, rec.fPzDeu, AliPID.ParticleMass(AliPID.kDeuteron))
        e_p = hypot4(rec.fPxP, rec.fPyP, rec.fPzP, AliPID.ParticleMass(AliPID.kProton))
        e_pi = hypot4(rec.fPxPi, rec.fPyPi, rec.fPzPi, AliPID.ParticleMass(AliPID.kPion))

        deu.SetPxPyPzE(rec.fPxDeu, rec.fPyDeu, rec.fPzDeu, e_deu)
        p.SetPxPyPzE(rec.fPxP, rec.fPyP, rec.fPzP, e_p)
        pi.SetPxPyPzE(rec.fPxPi, rec.fPyPi, rec.fPzPi, e_pi)

        hyp = deu + p + pi

        label = '{}_'.format(label_am[rec.fIsMatter]) + c_lab
        hist_rec[label].Fill(hyp.Pt())

input_file.Close()

output_file = TFile('~/3body_workspace/results/eff_test.root', 'recreate')

dict_hist_eff = {}

for lab in label_array:
    hist_eff = TH1D('fHistEfficiency_{}'.format(lab), '', n_bins, 0, 10)
    hist_eff.SetDirectory(0)

    for b in range(1, n_bins):
        count_sim = hist_sim[lab].Integral(b, b)

        count_rec = hist_rec[lab].Integral(b, b)

        eff = count_rec / count_sim
        err_eff = eff * (1 - eff) / count_sim

        hist_eff.SetBinContent(b, eff)
        hist_eff.SetBinError(b, err_eff)

    prp.histo_makeup(hist_eff, x_title='#it{p}_{T} (GeV/#it{c} )',
                     y_title='Efficiency #times Acceptance', color=prp.kRedC, y_range=(-0.01, 0.41), l_width=3)
    hist_eff.Write()

    dict_hist_eff[lab] = hist_eff


color_a = [prp.kTempsScale6, prp.kTempsScale5, prp.kTempsScale4]
color_m = [prp.kTempsScale0, prp.kTempsScale1, prp.kTempsScale2]

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

    h_a = dict_hist_eff[label_am[0] + '_' + label_centrality[cent-1]]
    h_a.GetYaxis().SetRangeUser(-0.12, 0.36)

    h_m = dict_hist_eff[label_am[1] + '_' + label_centrality[cent-1]]
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
