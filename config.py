#!/usr/bin/env python3
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--nbody', type=int, help='')
parser.add_argument('-d', '--download', action='store_true', help='')
args = vars(parser.parse_args())

NBODY = args['nbody']
DOWNLOAD_DATA = args['download']

if NBODY not in [2, 3]:
    print('2 or 3 body decay only in ALICE'), exit()

# check weather ROOT is available
if 'ROOTSYS' not in os.environ:
    print('ROOT required! Load it before executing this.'), exit()

# define variables
PWD = os.environ['PWD']
os.environ['HYPERML_DATA'] = PWD + '/Trees'
os.environ['HYPERML_TABLES'] = PWD +'/Tables'
os.environ['HYPERML_FIGURES'] = PWD +'/Figures'
os.environ['HYPERML_RESULTS'] = PWD +'/Results'
os.environ['HYPERML_MODELS'] = PWD + '/Models'

if DOWNLOAD_DATA:
    if 'ALICE_PHYSICS' not in os.environ:
        print('AliPhysics environment required! Load it for downloading data.'), exit()
