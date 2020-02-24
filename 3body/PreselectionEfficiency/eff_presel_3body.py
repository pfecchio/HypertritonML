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


def hypot3(x0, x1, x2):
    return math.sqrt(pot2(x0) + pot2(x1) + pot2(x2))


def hypot4(x0, x1, x2, x3):
    return math.sqrt(pot2(x0) + pot2(x1) + pot2(x2) + pot2(x3))


hyp_mass = 2.99131

suffix = '19d2'

# import environment
# input_file_path = '~/run_nitty/reference'
# input_file_path = '~/run_nitty/latest'
# input_file_path = '~/3body_workspace/preseleff_study/trees/'
input_file_path = os.environ['HYPERML_DATA_3']

# open input file and tree
input_file_name = f'newMC/HyperTritonTree_{suffix}.root'

input_file = TFile(f'{input_file_path}/{input_file_name}', 'read')

tree = input_file.fHypertritonTree
n_events = tree.GetEntries()

bins = np.array([1., 2., 4., 6., 8., 10., 14., 18., 23., 35., 50.])
# bins = np.arange(0., 50., 1.)
n_bins = len(bins) - 1

########################################
# ct sim and rec
hist_ctsim = TH1D('fHistCtSim', '', n_bins, bins)
hist_ctrec = TH1D('fHistCtRec', '', n_bins, bins)

hist_ctsim.SetDirectory(0)
hist_ctrec.SetDirectory(0)

########################################
# pt sim and rec
hist_ptsim = TH1D('fHistPtSim', '', 40, 0, 10)
hist_ptrec = TH1D('fHistPtRec', '', 40, 0, 10)

hist_ptsim.SetDirectory(0)
hist_ptrec.SetDirectory(0)

########################################
# p sim and rec
hist_psim = TH1D('fHistPSim', '', 40, 0, 10)
hist_prec = TH1D('fHistPRec', '', 40, 0, 10)

hist_psim.SetDirectory(0)
hist_prec.SetDirectory(0)

########################################
# eta sim e rec
hist_etasim = TH1D('fHistEtaSim', '', 40, -2, 2)
hist_etarec = TH1D('fHistEtaRec', '', 40, -2, 2)

hist_etasim.SetDirectory(0)
hist_etarec.SetDirectory(0)

########################################
# phi sim e rec
hist_phisim = TH1D('fHistPhiSim', '', 20, -5, 5)
hist_phirec = TH1D('fHistPhiRec', '', 20, -5, 5)

hist_phisim.SetDirectory(0)
hist_phirec.SetDirectory(0)

########################################
# l_sim-l_rec
hist_deltal = TH1D('fHistDeltaL', '', 400, -20, 20)
hist_deltal.SetDirectory(0)

########################################
# p_sim-p_rec
hist_deltap = TH1D('fHistDeltaP', '', 100, -5, 5)
hist_deltap.SetDirectory(0)

########################################
# eta_sim-eta_rec
hist_deltaeta = TH1D('fHistDeltaEta', '', 50, -2, 2)
hist_deltaeta.SetDirectory(0)

########################################
# ct_sim-ct_rec
hist_deltact = TH1D('fHistDeltaCt', '', 200, -10, 10)
hist_deltact.SetDirectory(0)

analyzed_events = 0
counter = 0

# main loop over the events
for ev in tree:
    if ev.REvent.fCent > 90.:
        continue

    # loop over the simulated hypertritons
    for sim in ev.SHypertriton:
        hyp = TLorentzVector()
        deu = TLorentzVector()
        p = TLorentzVector()
        pi = TLorentzVector()

        e_deu = hypot4(sim.fPxDeu, sim.fPyDeu, sim.fPzDeu, AliPID.ParticleMass(AliPID.kDeuteron))
        e_p = hypot4(sim.fPxP, sim.fPyP, sim.fPzP, AliPID.ParticleMass(AliPID.kProton))
        e_pi = hypot4(sim.fPxPi, sim.fPyPi, sim.fPzPi, AliPID.ParticleMass(AliPID.kPion))

        deu.SetXYZM(sim.fPxDeu, sim.fPyDeu, sim.fPzDeu, AliPID.ParticleMass(AliPID.kDeuteron))
        p.SetXYZM(sim.fPxP, sim.fPyP, sim.fPzP, AliPID.ParticleMass(AliPID.kProton))
        pi.SetXYZM(sim.fPxPi, sim.fPyPi, sim.fPzPi, AliPID.ParticleMass(AliPID.kPion))

        hyp = deu + p + pi

        decay_lenght = TVector3(sim.fDecayVtxX, sim.fDecayVtxY, sim.fDecayVtxZ)

        m = hyp_mass
        dl = decay_lenght.Mag()
        p = hyp.P()

        t = m * dl
        if hyp.Gamma() == 0 or hyp.Beta() == 0:
            continue
        ct = dl / (hyp.Gamma() * hyp.Beta())

        # if hyp.Pt() >= 1. or hyp.Pt() <= 10.:
        hist_ctsim.Fill(ct)
        hist_ptsim.Fill(hyp.Pt())
        hist_psim.Fill(hyp.P())
        hist_etasim.Fill(hyp.Eta())
        hist_phisim.Fill(hyp.Phi())

        # rec - sim diff
        if sim.fRecoIndex >= 0:
            r = ev.RHypertriton[sim.fRecoIndex]

            hyp_rec = TLorentzVector()
            deu_rec = TLorentzVector()
            p_rec = TLorentzVector()
            pi_rec = TLorentzVector()

            deu_rec.SetXYZM(r.fPxDeu, r.fPyDeu, r.fPzDeu, AliPID.ParticleMass(AliPID.kDeuteron))
            p_rec.SetXYZM(r.fPxP, r.fPyP, r.fPzP, AliPID.ParticleMass(AliPID.kProton))
            pi_rec.SetXYZM(r.fPxPi, r.fPyPi, r.fPzPi, AliPID.ParticleMass(AliPID.kPion))

            hyp_rec = deu_rec + p_rec + pi_rec

            p_rec = hyp_rec.P()
            decay_lenght_rec = TVector3(r.fDecayVtxX, r.fDecayVtxY, r.fDecayVtxZ)

            delta_l = decay_lenght.Mag() - decay_lenght_rec.Mag()
            delta_p = hyp.P() - hyp_rec.P()
            delta_eta = hyp.Eta() - hyp_rec.Eta()

            delta_ct = ct - (hyp_mass * decay_lenght_rec.Mag() / hyp_rec.P())

            hist_deltal.Fill(delta_l)
            hist_deltap.Fill(delta_p)
            hist_deltaeta.Fill(delta_eta)
            hist_deltact.Fill(delta_ct)

    # loop over the reconstructed hypertritons
    for rec in ev.RHypertriton:

        hyp = TLorentzVector()
        deu = TLorentzVector()
        p = TLorentzVector()
        pi = TLorentzVector()

        e_deu = hypot4(rec.fPxDeu, rec.fPyDeu, rec.fPzDeu, AliPID.ParticleMass(AliPID.kDeuteron))
        e_p = hypot4(rec.fPxP, rec.fPyP, rec.fPzP, AliPID.ParticleMass(AliPID.kProton))
        e_pi = hypot4(rec.fPxPi, rec.fPyPi, rec.fPzPi, AliPID.ParticleMass(AliPID.kPion))

        deu.SetXYZM(rec.fPxDeu, rec.fPyDeu, rec.fPzDeu, AliPID.ParticleMass(AliPID.kDeuteron))
        p.SetXYZM(rec.fPxP, rec.fPyP, rec.fPzP, AliPID.ParticleMass(AliPID.kProton))
        pi.SetXYZM(rec.fPxPi, rec.fPyPi, rec.fPzPi, AliPID.ParticleMass(AliPID.kPion))

        hyp = deu + p + pi

        decay_lenght = TVector3(rec.fDecayVtxX - ev.REvent.fX,
                                rec.fDecayVtxY - ev.REvent.fY, rec.fDecayVtxZ - ev.REvent.fZ)

        m = hyp_mass
        dl = decay_lenght.Mag()
        p = hyp.P()

        t = m * dl

        if hyp.Gamma() == 0 or hyp.Beta() == 0:
            continue
        ct = dl / (hyp.Gamma() * hyp.Beta())

        # if hyp.Pt() >= 1. or hyp.Pt() <= 10.:
        hist_ctrec.Fill(ct)
        hist_ptrec.Fill(hyp.Pt())
        hist_prec.Fill(hyp.P())
        hist_etarec.Fill(hyp.Eta())
        hist_phirec.Fill(hyp.Phi())

    analyzed_events += 1
    update_progress(analyzed_events/n_events)

input_file.Close()

# create output file
home_path = os.environ['HOME']
output_file_path = home_path + '/3body_workspace/preseleff_study'
output_file_name = f'PreselEff_{suffix}_newMC.root'

output_file = TFile(f'{output_file_path}/{output_file_name}', 'recreate')

hist_ctsim.Write()
hist_ctrec.Write()
hist_ptsim.Write()
hist_ptrec.Write()
hist_psim.Write()
hist_prec.Write()
hist_etasim.Write()
hist_etarec.Write()
hist_phisim.Write()
hist_phirec.Write()

# hist_deltal.Write()
# hist_deltap.Write()
# hist_deltaeta.Write()
# hist_deltact.Write()

################################################################################
# ct efficiency
################################################################################

hist_effct = hist_ctrec.Clone('fHistEfficiencyVsCt')
hist_effct.SetDirectory(0)

hist_effct.Divide(hist_ctsim)

for b in range(1, hist_effct.GetNbinsX() + 1):
    hist_effct.SetBinError(b, 0)

prp.histo_makeup(hist_effct, x_title='#it{c}t (cm)',
                 y_title='Efficiency #times Acceptance', color=prp.kBlueC, y_range=(-0.01, 0.8), l_width=2, opt='he')
hist_effct.Write()

################################################################################
# pt efficiency
################################################################################

hist_effpt = hist_ptrec.Clone('fHistEfficiencyVsPt')
hist_effpt.SetDirectory(0)

hist_effpt.Divide(hist_ptsim)

for b in range(1, hist_effpt.GetNbinsX() + 1):
    hist_effpt.SetBinError(b, 0)

prp.histo_makeup(hist_effpt, x_title='#it{p}_{T} (GeV/#it{c} )',
                 y_title='Efficiency #times Acceptance', color=prp.kBlueC, y_range=(-0.01, 0.8), l_width=2, opt='he')
hist_effpt.Write()

################################################################################
# p efficiency
################################################################################

hist_effp = hist_prec.Clone('fHistEfficiencyVsP')
hist_effp.SetDirectory(0)

hist_effp.Divide(hist_psim)

for b in range(1, hist_effp.GetNbinsX() + 1):
    hist_effp.SetBinError(b, 0)

prp.histo_makeup(hist_effp, x_title='#it{p} (GeV/#it{c} )',
                 y_title='Efficiency #times Acceptance', color=prp.kBlueC, y_range=(-0.01, 0.8), l_width=2, opt='he')
hist_effp.Write()

################################################################################
# eta efficiency
################################################################################

hist_effeta = hist_etarec.Clone('fHistEfficiencyVsEta')
hist_effeta.SetDirectory(0)

hist_effeta.Divide(hist_etasim)

for b in range(1, hist_effeta.GetNbinsX() + 1):
    hist_effeta.SetBinError(b, 0)

prp.histo_makeup(hist_effeta, x_title='#eta',
                 y_title='Efficiency #times Acceptance', color=prp.kBlueC, y_range=(-0.01, 0.8), l_width=2, opt='he')
hist_effeta.Write()

################################################################################
# phi efficiency
################################################################################

hist_effphi = hist_phirec.Clone('fHistEfficiencyVsPhi')
hist_effphi.SetDirectory(0)

hist_effphi.Divide(hist_phisim)

for b in range(1, hist_effphi.GetNbinsX() + 1):
    hist_effphi.SetBinError(b, 0)

prp.histo_makeup(hist_effphi, x_title='#phi',
                 y_title='Efficiency #times Acceptance', color=prp.kBlueC, y_range=(-0.01, 0.8), l_width=2, opt='he')
hist_effphi.Write()

output_file.Close()

print('')
