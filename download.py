#!/usr/bin/env python3
import argparse
import os

def scp_download(origin, destination):
    if not os.path.exists(file_name):
        os.system(f'scp {origin} {destination}')

################################################################################
# config options
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--nbody', type=int, help='Select decay channel <2|3>')
parser.add_argument('-p', '--pass', type=int, help='Select reco pass <1|3>')
parser.add_argument('-v', '--vzero', type=str, help='Select V0 for 2-body decay <"otf"|"off">')
parser.add_argument('-tab', '--tables', action='store_true', help='Download derived tables')
parser.add_argument('-tr', '--trees', action='store_true', help='Download trees')
args = vars(parser.parse_args())

################################################################################
# settings
NBODY = args['nbody']
PASS = args['pass']
TABLES = args['tables']
TREES = args['trees']
V0_FINDER = args['vzero'] if 'vzero' in args else 'off'
MC = '20g7' if PASS == 3 else '19d2'

if NBODY not in [2, 3]:
    print('2 or 3 body decay only in ALICE.'), exit()

if PASS not in [1, 3]:
    print('Reco pass1 and pass3 only avalilable.'), exit()

if V0_FINDER not in ['otf', 'off']:
    V0_FINDER = 'off'
    print('V0 finder options not valid. Setting default "offline"'), exit()

################################################################################
# check weather AliPhysics is available
if 'ALICE_PHYSICS' not in os.environ:
    print('AliPhysics required for downloading data. exit'), exit()

download_2body_trees = [[f'lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_18q_pass{PASS}{V0_FINDER}.root',
                        os.environ['HYPERML_DATA_2'] + f'HyperTritonTree_18r_pass{PASS}{V0_FINDER}.root'],
                        [f'lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_18q_pass{PASS}{V0_FINDER}.root',
                        os.environ['HYPERML_DATA_2'] + f'HyperTritonTree_18r_pass{PASS}{V0_FINDER}.root'],
                        [f'lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_{MC}{V0_FINDER}.root',
                        os.environ['HYPERML_DATA_2'] + f'HyperTritonTree_{MC}{V0_FINDER}.root'],
                        [f'lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_18qLS_pass{PASS}.root',
                        os.environ['HYPERML_DATA_2'] + f'/HyperTritonTree_18qLS_pass{PASS}.root'],
                        [f'lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_18rLS_pass{PASS}.root',
                        os.environ['HYPERML_DATA_2'] + f'/HyperTritonTree_18rLS_pass{PASS}.root']]
download_2body_tables = []
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
if NBODY == 2:

    for source, dest in download_2body_utils:
        scp_download(source, dest)

