#!/usr/bin/env python3
import argparse
import os
from pathlib import Path 

def scp_download(origin, destination):
    if not os.path.exists(destination):
        print(f'Downloading "{Path(origin).name}" into "{destination}"')
        os.system(f'scp {origin} {destination}')

################################################################################
# config options
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--nbody', type=int, help='Select decay channel <2|3>')
parser.add_argument('-p', '--pass', type=int, help='Select reco pass <1|3>')
parser.add_argument('-v', '--vzero', type=str, help='Select V0 for 2-body decay <"otf"|"off">')
parser.add_argument('-tb', '--tables', action='store_true', help='Download derived tables')
parser.add_argument('-tr', '--trees', action='store_true', help='Download trees')
parser.add_argument('-u', '--utils', action='store_true', help='Download utils')
args = vars(parser.parse_args())

################################################################################
# settings
NBODY = args['nbody']
PASS = args['pass']
TABLES = args['tables']
TREES = args['trees']
UTILS = args['utils']
V0_FINDER = ''
if args['vzero'] == 'otf':
    V0_FINDER = '_otf'

if NBODY not in [2, 3, None]:
    print('2 or 3 body decay only in ALICE.'), exit()
if NBODY is None:
    print('Select a decay channel.'), exit()

if PASS not in [1, 3, None]:
    print('Reco pass1 and pass3 only avalilable.'), exit()
if PASS is None:
    PASS = 3
    print('Reeco pass not selected, setting default "pass3".')

if not TREES and not TABLES:
    print('No option for trees or tables, downloading both.')
    TREES = True
    TABLES = True

if NBODY is 2:
    if args['vzero'] not in ['otf', 'off']:
        print('V0 finder option not valid.'), exit()
    if args['vzero'] is None:
        print('V0 finder not selected, setting default "offline".')

MC = '20g7' if PASS == 3 else '19d2'

# ################################################################################
# downlaod lists
download_2body_trees = [[f'lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_18q_pass{PASS}{V0_FINDER}.root',
                        os.environ['HYPERML_TREES_2'] + f'HyperTritonTree_18r_pass{PASS}{V0_FINDER}.root'],
                        [f'lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_18q_pass{PASS}{V0_FINDER}.root',
                        os.environ['HYPERML_TREES_2'] + f'HyperTritonTree_18r_pass{PASS}{V0_FINDER}.root'],
                        [f'lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_{MC}{V0_FINDER}.root',
                        os.environ['HYPERML_TREES_2'] + f'HyperTritonTree_{MC}{V0_FINDER}.root'],
                        [f'lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_18qLS_pass{PASS}.root',
                        os.environ['HYPERML_TREES_2'] + f'/HyperTritonTree_18qLS_pass{PASS}.root'],
                        [f'lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_18rLS_pass{PASS}.root',
                        os.environ['HYPERML_TREES_2'] + f'/HyperTritonTree_18rLS_pass{PASS}.root']]

download_2body_tables = [[f'lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/Tables/DataTable_18_pass{PASS}{V0_FINDER}.root',
                        os.environ['HYPERML_TABLES_2'] + f'DataTable_18_pass{PASS}{V0_FINDER}.root'],
                        [f'lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/Tables/DataTable_18LS_pass{PASS}{V0_FINDER}.root',
                        os.environ['HYPERML_TABLES_2'] + f'DataTable_18LS_pass{PASS}{V0_FINDER}.root'],
                        [f'lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/Tables/SignalTable_{MC{V0_FINDER}.root',
                        os.environ['HYPERML_TABLES_2'] + f'SignalTable_{MC}{V0_FINDER}.root']]

download_2body_utils = [['lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/Utils/BlastWaveFits.root',
                        os.environ['HYPERML_UTILS'] + f'/BlastWaveFit.root'],
                        ['lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/Utils/He3TPCCalibration.root',
                        os.environ['HYPERML_UTILS'] + f'/He3TPCCalibration.root'],
                        ['lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/Utils/AbsorptionHe3.root',
                        os.environ['HYPERML_UTILS'] + f'/AbsorptionHe3.root']]

download_3body_trees = []
download_3body_tables = []
download_3body_utils = []

################################################################################
# finally download
if NBODY == 2:
    if UTILS:
        for source, dest in download_2body_utils:
            scp_download(source, dest)
    
    if TREES:
        for source, dest in download_2body_utils:
            scp_download(source, dest)

    if TABLES:
        for source, dest in download_2body_utils:
            scp_download(source, dest)


# if NBODY == 3:
#     if UTILS:
#         for source, dest in download_3body_utils:
#             scp_download(source, dest)
    
#     if TREES:
#         for source, dest in download_3body_utils:
#             scp_download(source, dest)

#     if TABLES:
#         for source, dest in download_3body_utils:
#             scp_download(source, dest)
