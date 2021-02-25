#!/usr/bin/env python3
import argparse
import os


################################################################################
# config options
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--nbody', type=int, help='')
parser.add_argument('-p', '--pass', type=int, help='')
args = vars(parser.parse_args())

NBODY = args['nbody']
PASS = args['pass']

if NBODY not in [2, 3]:
    print('2 or 3 body decay only in ALICE.'), exit()

if PASS not in [1, 3]:
    print('Reco pass1 and pass3 only avalilable.'), exit()

################################################################################
# check weather ROOT is available
if os.environ['ALICE']


################################################################################
# set environment variables and create dirs
dir_list = ['Trees', 'Tables', 'Figures', 'Results', 'Models', 'Efficiencies', 'Utils'] 
base_dir = os.environ['PWD']

for d in dir_list:
    dir_path = base_dir + f'/{d}/{NBODY}Body'
    os.environ[f'HYPERML_{d.upper()}_{NBODY}'] = dir_path

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

os.environ['HYPERML_UTILS'] = base_dir + '/Utils'
os.environ['HYPERML_CODE'] = base_dir + '/common/TrainingAndTesting'

 
################################################################################
# download data if required
if DOWNLOAD_DATA:
    if 'ALICE_PHYSICS' not in os.environ:
        print('AliPhysics environment required! Load it for downloading data.'), exit()

    if NBODY is 2:
        file_name = os.environ['HYPERML_UTILS'] + '/BlastWaveFit.root'
        if not os.path.exists(file_name):
            os.system(f'alien_cp /alice/cern.ch/user/m/mpuccio/hyper_data/BlastWaveFits.root file://{file_name}')

        file_name = os.environ['HYPERML_TREES_2BODY'] + '/HyperTritonTree_18q_pass3.root'
        if not os.path.exists(file_name):
            os.system(f'scp lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_18q_pass3.root {file_name}')

        file_name = os.environ['HYPERML_TREES_2BODY'] + '/HyperTritonTree_18r_pass3.root'
        if not os.path.exists(file_name):
            os.system(f'scp lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_18r_pass3.root {file_name}')

        file_name = os.environ['HYPERML_TREES_2BODY'] + '/HyperTritonTree_18qLS_pass3.root'
        if not os.path.exists(file_name):
            os.system(f'scp lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_18qLS_pass3.root {file_name}')

        file_name = os.environ['HYPERML_TREES_2BODY'] + '/HyperTritonTree_18rLS_pass3.root'
        if not os.path.exists(file_name):
            os.system(f'scp lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_18rLS_pass3.root {file_name}')

        file_name = os.environ['HYPERML_TREES_2BODY'] + '/HyperTritonTree_18q_pass3_otf.root'
        if not os.path.exists(file_name):
            os.system(f'scp lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_18q_pass3_otf.root {file_name}')

        file_name = os.environ['HYPERML_TREES_2BODY'] + '/HyperTritonTree_18r_pass3_otf.root'
        if not os.path.exists(file_name):
            os.system(f'scp lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_18r_pass3_otf.root {file_name}')

        file_name = os.environ['HYPERML_TREES_2BODY'] + '/HyperTritonTree_20g7.root'
        if not os.path.exists(file_name):
            os.system(f'scp lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_20g7.root {file_name}')

        file_name = os.environ['HYPERML_TREES_2BODY'] + '/HyperTritonTree_20g7_otf.root'
        if not os.path.exists(file_name):
            os.system(f'scp lxplus.cern.ch:/eos/user/h/hypertriton/trees/2Body/HyperTritonTree_20g7_otf.root {file_name}')

        file_name = os.environ['HYPERML_UTILS'] + '/He3TPCCalibration.root'
        if not os.path.exists(file_name):
            os.system(f'alien_cp /alice/cern.ch/user/m/mpuccio/hyper_data/He3TPCCalibration.root file://{file_name}')

        file_name = os.environ['HYPERML_UTILS'] + '/AbsorptionHe3.root'
        if not os.path.exists(file_name):
            os.system(f'alien_cp /alice/cern.ch/user/m/mpuccio/hyper_data/AbsorptionHe3.root file://{file_name}')
