import training_utils as tu
import os
import pickle

params_def = {
    # Parameters that we are going to tune.
    'max_depth':8,
    'eta':0.05,
    'gamma':0.7,
    'min_child_weight':8,
    'subsample':0.8,
    'colsample_bytree':0.9,
    'objective':'binary:logistic',
    'random_state':42,
    'silent':1,
    'nthread':4,
    'tree_method':'hist',
    'scale_pos_weight': 10}

training_columns = [ 'V0CosPA','ProngsDCA', 'DistOverP','ArmenterosAlpha','NpidClustersHe3','V0pt','TPCnSigmaHe3','He3ProngPvDCA','PiProngPvDCA']
print('1')
Analysis = tu.Generalized_Analysis('../DerivedTrees/SignalTable.root','../DerivedTrees/DataTable.root','ProngsDCA<1.6 and He3ProngPvDCA>0.01 and He3ProngPvDCA>0.01 and V0CosPA>0.98 and Centrality<10','(InvMass<2.98 or InvMass>3.005) and V0pt<=10')
print('2')
model =Analysis.TrainingAndTest(training_columns,params_def,draw=False)
print('3')
#v = Analysis.Significance(model,training_columns)
#print(v)

# loop over all the bins
# if not os.path.exists('Models'):
#   os.makedirs('Models')
# Centrality_bins = [[0,10],[10,30],[30,50],[50,90]]
# Pt_bins = [[2,3],[3,4],[4,5],[5,9]]
# Ct_bins = [[0,2],[2,4],[4,6],[6,8],[8,10],[10,14],[14,18],[18,23],[23,28]]
# for index_cen in range(0,len(Centrality_bins)):
#   for index_pt in range(0,len(Pt_bins)):
#     for index_ct in range(0,len(Ct_bins)):
#       model =Analysis.TrainingAndTest(training_columns,params_def,Ct_bins[index_ct],Pt_bins[index_pt],Centrality_bins[index_cen],draw=False)
#       filename = 'Models/BDT_Ct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_Cen_{:.2f}_{:.2f}.sav'.format(Ct_bins[index_ct][0],Ct_bins[index_ct][1],Pt_bins[index_pt][0],Pt_bins[index_pt][1],Centrality_bins[index_cen][0],Centrality_bins[index_cen][1])
#       pickle.dump(model, open(filename, 'wb'))
#       print(filename+' has been saved')
