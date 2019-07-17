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

output_file = TFile('~/3body_workspace/results/eff_hist.root', 'recreate')

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
