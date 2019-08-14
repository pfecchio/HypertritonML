#!/bin/bash
export HYPERML_DATA="$HOME/HypertritonAnalysis/Trees"
export HYPERML_TABLES="$HOME/HypertritonAnalysis/DerivedTrees"
export HYPERML_FIGURES="$HOME/HypertritonAnalysis/Figures"
export HYPERML_MODELS="$HOME/HypertritonAnalysis/Models"
[ ! -d "$HYPERML_DATA" ] && mkdir -p $HYPERML_DATA
[ ! -f "$HYPERML_DATA/HyperTritonTree_18q.root"  ] && alien_cp alien:/alice/cern.ch/user/f/fmazzasc/Trees/HyperTritonTree_18q.root  $HYPERML_DATA/HyperTritonTree_18q.root
[ ! -f "$HYPERML_DATA/HyperTritonTree_18r.root"  ] && alien_cp alien:/alice/cern.ch/user/f/fmazzasc/Trees/HyperTritonTree_18r.root  $HYPERML_DATA/HyperTritonTree_18r.root
[ ! -f "$HYPERML_DATA/HyperTritonTree_19d2.root" ] && alien_cp alien:/alice/cern.ch/user/f/fmazzasc/Trees/HyperTritonTree_19d2.root $HYPERML_DATA/HyperTritonTree_19d2.root

[ ! -d "$HYPERML_TABLES" ] && mkdir -p $HYPERML_TABLES
