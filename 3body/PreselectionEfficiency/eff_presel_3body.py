import math
import os
import sys
import time

import numpy as np

import pyroot_plot as prp
from ROOT import (TH1D, TH2D, AliPID, TCanvas, TFile, TGaxis, TLegend,
                  TLorentzVector, TPad, TTree, TVector3, gROOT)


# usefull progressbar
# update_progress() : Displays or updates a console progress bar
def update_progress(progress):
    barLength = 40  # Modify this to change the length of the progress bar
    status = ''
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = 'error: progress var must be float\r\n'
    if progress < 0:
        progress = 0
        status = 'Halt...\r\n'
    if progress >= 1:
        progress = 1
        status = 'Done...\r\n'
    block = int(round(barLength*progress))
    text = '\rPercent: [{0}] {1:g}% {2}'.format('#'*block + '-'*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


# useful methods
def pot2(x):
    return x*x


def hypot4(x0, x1, x2, x3):
    return math.sqrt(pot2(x0) + pot2(x1) + pot2(x2) + pot2(x3))


# import environment
# input_file_path = '~/run_nitty/reference'
input_file_path = '~/run_nitty/latest'

# open input file and tree
input_file_name = 'HyperTritonTree.root'

input_file = TFile(f'{input_file_path}/{input_file_name}', 'read')

tree = input_file.fHypertritonTree
n_events = tree.GetEntries()

N_BINS = 40

# create histos for the efficiency
hist_sim = TH1D('fHistSim', '', 35, 0, 35)
hist_rec = TH1D('fHistRec', '', 35, 0, 35)

hist_sim.SetDirectory(0)
hist_rec.SetDirectory(0)

hist_costheta_sim = TH1D('fHistCosThetaSim', '', 1000, -0.01, 1.)
hist_costheta_rec = TH1D('fHistCosThetaRec', '', 1000, -0.01, 1.)

hist_costheta_sim.SetDirectory(0)
hist_costheta_rec.SetDirectory(0)

analyzed_events = 0
counter = 0

# main loop over the events
for ev in tree:
    if counter > 10000:
        break

    counter = counter + 1

    # loop over the simulated hypertritons
    for sim in ev.SHypertriton:

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

        # decay_lenght = decay_vtx - primary_vtx
        dl = [sim.fDecayVtxX, sim.fDecayVtxY, sim.fDecayVtxZ]
        dl_norm = math.sqrt(dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2])

        cos_theta = hyp.Px() * dl[0] + hyp.Py() * dl[1] + hyp.Pz() * dl[2]
        cos_theta /= dl_norm * hyp.P()

        ct = 2.99131 * dl_norm / hyp.P()

        hist_sim.Fill(ct)
        hist_costheta_sim.Fill(cos_theta)

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

        dl = [rec.fDecayVtxX - ev.REvent.fX, rec.fDecayVtxY - ev.REvent.fY, rec.fDecayVtxZ - ev.REvent.fZ]
        dl_norm = math.sqrt(dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2])

        cos_theta = hyp.Px() * dl[0] + hyp.Py() * dl[1] + hyp.Pz() * dl[2]
        cos_theta /= dl_norm * hyp.P()

        ct = 2.99131 * dl_norm / hyp.P()

        hist_rec.Fill(ct)
        hist_costheta_rec.Fill(cos_theta)

    analyzed_events += 1
    update_progress(analyzed_events/n_events)

input_file.Close()

# create output file
home_path = os.environ['HOME']
output_file_path = home_path + '/HypertritonAnalysis/PreselEfficiency/3Body'
output_file_name = 'PreselectionEfficiencyHist.root'

output_file = TFile(output_file_name, 'recreate')

hist_sim.Write()
hist_rec.Write()
hist_costheta_sim.Write()
hist_costheta_rec.Write()

# compute efficiency
CT_BINS = 9
bins = np.array([1., 2., 4., 6., 8., 10., 14., 18., 23., 35.])

hist_eff = TH1D('fHistEfficiencyVsCt', '', CT_BINS, bins)
hist_eff.SetDirectory(0)

for b in range(1, CT_BINS+1):
    count_sim = hist_sim.Integral(b, b)
    count_rec = hist_rec.Integral(b, b)

    if count_sim != 0:
        eff = count_rec / count_sim
        err_eff = eff * (1 - eff) / count_sim
    else:
        eff = 0
        err_eff = 0

    hist_eff.SetBinContent(b, eff)
    hist_eff.SetBinError(b, err_eff)

prp.histo_makeup(hist_eff, x_title='#it{c}t (cm)',
                 y_title='Efficiency #times Acceptance', color=prp.kRedC, y_range=(-0.01, 0.8), l_width=3)
hist_eff.Write()

output_file.Close()

print('')
