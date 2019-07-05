import os

from ROOT import TH3, TFile, TList

label = ['a1', 'a2', 'b', 'c']

for l in label:
    input_file = TFile('~/data/3body_hypetriton_data/HyperFindable3{}.root'.format(l), 'read')
    summary_list = input_file.Get('Hyp3FindTask_summary')

    hist_ptycent_hyp = summary_list.FindObject('fHistGeneratedPtVsYVsCentralityHypTrit')
    hist_ptycent_antihyp = summary_list.FindObject('fHistGeneratedPtVsYVsCentralityAntiHypTrit')

    hist_ptycent_hyp.SetDirectory(0)
    hist_ptycent_antihyp.SetDirectory(0)

    input_file.Close()

    output_file = TFile('~/3body_workspace/results/MCinfo{}.root'.format(l), 'recreate')

    hist_ptycent_hyp.Write()
    hist_ptycent_antihyp.Write()

    output_file.Close()
