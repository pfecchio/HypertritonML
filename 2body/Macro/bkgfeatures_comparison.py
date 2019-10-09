#!/usr/bin/env python3
import os

import pandas as pd
import plot_utils as pu
import uproot

ls_path = os.path.expandvars('$HYPERML_TABLES_2/DataTableLS.root')
data_path = os.path.expandvars('$HYPERML_TABLES_2/DataTable.root')

ls_selection = '2<=HypCandPt<=10'
data_selection = '(InvMass<2.98 or InvMass>3.005) and 2<=HypCandPt<=10'

df_ls = uproot.open(ls_path)['DataTable'].pandas.df().query(ls_selection)
df_data = uproot.open(data_path)['DataTable'].pandas.df().query(data_selection)

df_ls['y'] = 0
df_data['y'] = 1

training_columns = ['V0CosPA',
                    'HypCandPt',
                    'ProngsDCA',
                    'PiProngPvDCAXY',
                    'He3ProngPvDCAXY',
                    'He3ProngPvDCA',
                    'PiProngPvDCA',
                    'NpidClustersHe3',
                    'TPCnSigmaHe3']

df = pd.concat([df_ls, df_data])

pu.plot_distr(df, column=training_columns, mode=2)

print('Background distribution comparison done!')
