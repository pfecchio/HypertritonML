#!/usr/bin/env python3

import os
from ROOT import gROOT

gROOT.SetBatch(True)

gROOT.LoadMacro("GenerateTableFromMC.cc")
gROOT.LoadMacro("GenerateTableFromData.cc")
from ROOT import GenerateTableFromMC, GenerateTableFromData

input_dir = os.environ['HYPERML_DATA_2']+ "/splines_trees"
output_dir = os.environ['HYPERML_TABLES_2']+ "/kf_tables"

print("++++++++++++++++++++++++++++++++++++++++++")
print("Generate Signal Table")
print("++++++++++++++++++++++++++++++++++++++++++")
GenerateTableFromMC(True, input_dir, output_dir)
print("++++++++++++++++++++++++++++++++++++++++++")
print("Generate Data Table")
print("++++++++++++++++++++++++++++++++++++++++++")
GenerateTableFromData(False, False , input_dir, output_dir)
print("++++++++++++++++++++++++++++++++++++++++++")
print("Generate Like-Sign Backgoundd Table")
print("++++++++++++++++++++++++++++++++++++++++++")
GenerateTableFromData(True, False , input_dir ,output_dir)
