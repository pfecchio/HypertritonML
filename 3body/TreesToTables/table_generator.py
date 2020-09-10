#!/usr/bin/env python3

import os
from ROOT import gROOT

gROOT.SetBatch(True)


gROOT.LoadMacro("PrepareDataFrames.cc")
from ROOT import PrepareDataFrames

input_dir = "/../../../../../../data01/fmazzasc/merge"
output_dir = "/../../../../../../data01/fmazzasc/merge"

# print("++++++++++++++++++++++++++++++++++++++++++")
# print("Generate Signal Table")
# print("++++++++++++++++++++++++++++++++++++++++++")
# GenerateTableO2(True, input_dir + "/HyperTritonTree_19d2.root", + "/SignalTable_19d2.root")
print("++++++++++++++++++++++++++++++++++++++++++")
print("Generate Data Table")
print("++++++++++++++++++++++++++++++++++++++++++")
PrepareDataFrames("data", "", output_dir)
print("++++++++++++++++++++++++++++++++++++++++++")
print("Generate Like-Sign Backgoundd Table")
print("++++++++++++++++++++++++++++++++++++++++++")
PrepareDataFrames("LS", "", output_dir)
