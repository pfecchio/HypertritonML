import math
import os
import sys
import time

import pyroot_plot as prp
from ROOT import (TH1D, AliPID, TCanvas, TFile, TGaxis, TLegend,
                  TLorentzVector, TPad, TTree, gROOT)


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


n_bins = 20
pt_bin_width = float(10 / n_bins)

# import environment
input_file_path = os.environ['HYPERML_DATA_3']

# labels
label_centrality = ['0-10', '10-30', '30-50', '50-90']
label_am = ['antihyper', 'hyper']

# open input file and tree
input_file_name = 'HyperTritonTree_19d2.root'

input_file = TFile('{}/{}'.format(input_file_path, input_file_name), 'read')

tree = input_file.fHypertritonTree
n_events = tree.GetEntries()

# create histos for the efficiency
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

analyzed_events = 0

counter = 0

# main loop over the events
for ev in tree:
    # if counter > 10000:
    #     break

    # counter = counter + 1

    centrality = ev.REvent.fCent

    c_lab = ''

    if centrality <= 10.:
        c_lab = '{}'.format(label_centrality[0])

    elif centrality <= 30.:
        c_lab = '{}'.format(label_centrality[1])

    elif centrality <= 50.:
        c_lab = '{}'.format(label_centrality[2])

    elif centrality <= 90.:
        c_lab = '{}'.format(label_centrality[3])

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

    analyzed_events += 1
    update_progress(analyzed_events/n_events)

input_file.Close()

# create output file
home_path = os.environ['HOME']
output_file_path = home_path + '/HypertritonAnalysis/PreselEfficiency/3Body'
output_file_name = 'PreselectionEfficiencyHist.root'

output_file = TFile('{}/{}'.format(output_file_path, output_file_name), 'recreate')
output_file_txt = open('eff.txt', 'w')

# dictionary to manage histos
dict_hist_eff = {}

# compute efficiency
for lab in label_array:
    hist_eff = TH1D('fHistEfficiency_{}'.format(lab), '', n_bins, 0, 10)
    hist_eff.SetDirectory(0)

    output_file_txt.write('-- efficiency {} -- \n'.format(lab))

    for b in range(1, n_bins+1):
        count_sim = hist_sim[lab].Integral(b, b)
        count_rec = hist_rec[lab].Integral(b, b)

        eff = count_rec / count_sim
        err_eff = eff * (1 - eff) / count_sim

        hist_eff.SetBinContent(b, eff)
        hist_eff.SetBinError(b, err_eff)

        pt_bin = [(b - 1) * pt_bin_width, b * pt_bin_width]

        output_file_txt.write('{:.1f} - {:.1f}    {:.4f} +- {:.4f} \n'.format(pt_bin[0], pt_bin[1], eff, err_eff))

    output_file_txt.write('\n')

    prp.histo_makeup(hist_eff, x_title='#it{p}_{T} (GeV/#it{c} )',
                     y_title='Efficiency #times Acceptance', color=prp.kRedC, y_range=(-0.01, 0.41), l_width=3)
    hist_eff.Write()

output_file.Close()
output_file_txt.close()

os.system('mv eff.txt {}/eff.txt'.format(output_file_path))
