#!/usr/bin/env python3

import os
from ROOT import gROOT

gROOT.SetBatch(True)

gROOT.LoadMacro("GenerateTableFromMC.cc")
gROOT.LoadMacro("GenerateTableFromData.cc")
from ROOT import GenerateTableFromMC

input_dir = "/data/fmazzasc/PbPb_2body/trees"
output_dir = os.environ['HYPERML_TABLES_2']
print(output_dir)

print("++++++++++++++++++++++++++++++++++++++++++")
print("Generate Signal Table")
print("++++++++++++++++++++++++++++++++++++++++++")
GenerateTableFromMC(True, input_dir, output_dir)
