import math

import pyroot_plot as prp
from ROOT import TH1F, TH2, TH3, TCanvas, TFile, TLegend

###############################################################################
#                                                                             #
#       macro efficiency end MC studies for the hypertriton 3 body decay      #
#                                                                             #
###############################################################################

label_am = ('A', 'M')
label_cuts = ('findable', 'cuts', 'hard')
label_stat = ('', '_reflection', '_fake')

label = tuple([a, b] for a in label_cuts for b in label_stat)

# get the reference for the efficiency
input_file_ref = TFile('~/3body_workspace/results/MCinfo_tot.root', 'read')

hist_ptycent_hyp = input_file_ref.Get('fHistGeneratedPtVsYVsCentralityHypTrit')
hist_ptycent_antihyp = input_file_ref.Get('fHistGeneratedPtVsYVsCentralityAntiHypTrit')
hist_ptycent_hyp.SetDirectory(0)
hist_ptycent_antihyp.SetDirectory(0)

input_file_ref.Close()

hist_pt_hyp = hist_ptycent_hyp.Project3D('x')
hist_pt_antihyp = hist_ptycent_antihyp.Project3D('x')

bin_map = [[n * 10 + 1, (n + 1) * 10] for n in range(20)]

ref_hyp = []
ref_antihyp = []

for pt_min, pt_max in bin_map:
    ref_hyp.append(hist_pt_hyp.Integral(pt_min, pt_max) / 2.)
    ref_antihyp.append(hist_pt_antihyp.Integral(pt_min, pt_max) / 2.)

ref_tot = [x + y for x, y in zip(ref_hyp, ref_antihyp)]

#-----------------------------------------------------------------------------#

# get the reconstructed (anti-)hypertritons
input_file = TFile('~/3body_workspace/output/selector_output_tot_hard.root', 'read')

bin_map = [[n + 1, (n + 1)] for n in range(20)]

hist_recmap_antihyp = {}
hist_recmap_hyp = {}
hist_recmap_tot = {}

hist_array_antihyp = []
hist_array_hyp = []

for cuts, stat in label:
    hist_tmp_antihyp = input_file.Get('fHistInvMass_A_{}{}'.format(cuts, stat))
    hist_tmp_antihyp.SetName('antihyp_{}{}'.format(cuts, stat))
    hist_tmp_hyp = input_file.Get('fHistInvMass_M_{}{}'.format(cuts, stat))
    hist_tmp_hyp.SetName('hyp_{}{}'.format(cuts, stat))

    count_antihyp = []
    count_hyp = []

    for pt_min, pt_max in bin_map:
        count_antihyp.append(hist_tmp_antihyp.Integral(1, 200, pt_min, pt_max))
        count_hyp.append(hist_tmp_hyp.Integral(1, 200, pt_min, pt_max))

    hist_recmap_antihyp[hist_tmp_antihyp.GetName()] = count_antihyp
    hist_recmap_hyp[hist_tmp_hyp.GetName()] = count_hyp
    hist_recmap_tot[hist_tmp_antihyp.GetName().replace('antihyp', 'tot')] = [
        x + y for x, y in zip(count_antihyp, count_hyp)]

input_file.Close()

#-----------------------------------------------------------------------------#
# compute the efficiency and make the histos

hist_eff = {}
hist_eff_label = []

output_file = TFile('~/3body_workspace/results/selector_eff_tot_hard.root', 'recreate')

for l in label_cuts:
    eff_antihyp_tmp = TH1F('eff_antihyp_{}'.format(l), '', 20, 0, 10)
    eff_hyp_tmp = TH1F('eff_hyp_{}'.format(l), '', 20, 0, 10)
    eff_tot_tmp = TH1F('eff_tot_{}'.format(l), '', 20, 0, 10)

    for i in range(0, 20):
        eff_antihyp = hist_recmap_antihyp['antihyp_{}{}'.format(l, '')][i] / ref_antihyp[i]
        eff_hyp = hist_recmap_hyp['hyp_{}{}'.format(l, '')][i] / ref_hyp[i]
        eff_tot = hist_recmap_tot['tot_{}{}'.format(l, '')][i] / ref_tot[i]

        err_eff_antihyp = math.sqrt(eff_antihyp * (1-eff_antihyp) / ref_antihyp[i])
        err_eff_hyp = math.sqrt(eff_hyp * (1-eff_hyp) / ref_hyp[i])
        err_eff_tot = math.sqrt(eff_tot * (1-eff_tot) / ref_tot[i])

        eff_antihyp_tmp.SetBinContent(i+1, eff_antihyp)
        eff_hyp_tmp.SetBinContent(i+1, eff_hyp)
        eff_tot_tmp.SetBinContent(i+1, eff_tot)

        eff_antihyp_tmp.SetBinError(i+1, err_eff_antihyp)
        eff_hyp_tmp.SetBinError(i+1, err_eff_hyp)
        eff_tot_tmp.SetBinError(i+1, err_eff_tot)

    label_eff_antihyp = 'antihyp_{}{}'.format(l, '')
    label_eff_hyp = 'hyp_{}{}'.format(l, '')
    label_eff_tot = 'tot_{}{}'.format(l, '')

    hist_eff[label_eff_antihyp] = eff_antihyp_tmp
    hist_eff[label_eff_hyp] = eff_hyp_tmp
    hist_eff[label_eff_tot] = eff_tot_tmp

    hist_eff_label.append(label_eff_antihyp)
    hist_eff_label.append(label_eff_hyp)
    hist_eff_label.append(label_eff_tot)

#-----------------------------------------------------------------------------#
# histo make-up
#-----------------------------------------------------------------------------#

for l in hist_eff_label:
    h = hist_eff[l]
    color = prp.kBlueC

    if 'anti' in l:
        color = prp.kRedC

    prp.histo_makeup(h, x_title='#it{p}_{T} (GeV/#it{c} )',
                     y_title='Efficiency #times Acceptance', color=color, y_range=(-0.01, 0.61))
    h.Write()

#-----------------------------------------------------------------------------#
# anti-matter/matter ratios
#-----------------------------------------------------------------------------#

for l in label_cuts:
    legend = TLegend(0.165, 0.650, 0.474, 0.865, 'C')
    legend.SetFillStyle(0)
    legend.SetTextSize(18)
    legend.SetHeader('')
    legend.AddEntry(hist_eff['hyp_{}'.format(l)], 'hyp_{}'.format(l))
    legend.AddEntry(hist_eff['antihyp_{}'.format(l)], 'antihyp_{}'.format(l))

    prp.ratio_plot(h1=hist_eff['hyp_{}'.format(l)], h2=hist_eff['antihyp_{}'.format(l)], dim=(
        700, 600), mode='ratio', l_range=(0.78, 1.04), l_y_title='antihyp/hyp ratio', legend=legend)

#-----------------------------------------------------------------------------#
# cuts comparison
#-----------------------------------------------------------------------------#

c = TCanvas('tot_selec_comparison', '', 700, 500)

legend = TLegend(0.165, 0.650, 0.474, 0.865, 'C')
legend.SetFillStyle(0)
legend.SetTextSize(18)
legend.SetHeader('selection comparison')

color_index = 0

for l in label_cuts:
    legend.AddEntry(hist_eff['tot_{}'.format(l)], 'tot_{}'.format(l))

    h = hist_eff['tot_{}'.format(l)]
    prp.histo_makeup(h, x_title='#it{p}_{T} (GeV/#it{c} )',
                     y_title='Efficiency #times Acceptance', color=prp.colors[color_index], y_range=(-0.01, 0.91))

    opt = ''
    if color > 0:
        opt = opt + 'same'
    h.Draw(opt)
    color_index = color_index + 1

legend.Draw()
c.Write()

# output_file.Write()
output_file.Close()
