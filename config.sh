#!/bin/bash
export HYPERML_DATA="$HOME/HypertritonAnalysis/Trees"
export HYPERML_TABLES="$HOME/HypertritonAnalysis/DerivedTrees"

BODY_2=0
BODY_3=0

if [ $# -eq 0 ]; then
      BODY_2=1
      BODY_3=1
fi

if [ "$1" == "2" ] || [ "$1" == "2body" ] || [ "$1" == "2Body" ]; then
      BODY_2=1
fi
if [ "$1" == "3" ] || [ "$1" == "3body" ] || [ "$1" == "3Body" ]; then
      BODY_3=1
fi


if [ $BODY_2 -eq 1 ]; then    
      export HYPERML_DATA_2="$HYPERML_DATA/2Body"
      export HYPERML_TABLES_2="$HYPERML_TABLES/2Body"

      [ ! -d "$HYPERML_DATA_2" ] && mkdir -p $HYPERML_DATA_2
      [ ! -f "$HYPERML_DATA_2/HyperTritonTree_18q.root"  ] &&alien:/alice/cern.ch/user/f/fmazzasc/Trees/HyperTritonTree_18q.root  $HYPERML_DATA_2/HyperTritonTree_18q.root
      [ ! -f "$HYPERML_DATA_2/HyperTritonTree_18r.root"  ] &&alien:/alice/cern.ch/user/f/fmazzasc/Trees/HyperTritonTree_18r.root  $HYPERML_DATA_2/HyperTritonTree_18r.root
      [ ! -f "$HYPERML_DATA_2/HyperTritonTree_19d2.root" ] &&alien:/alice/cern.ch/user/f/fmazzasc/Trees/HyperTritonTree_19d2.root $HYPERML_DATA_2/HyperTritonTree_19d2.root

      [ ! -d "$HYPERML_TABLES_2" ] && mkdir -p $HYPERML_TABLES_2
fi

if [ $BODY_3 -eq 1 ]; then    
      export HYPERML_DATA_3="$HYPERML_DATA/3Body"
      export HYPERML_TABLES_3="$HYPERML_TABLES/3Body"

      [ ! -d "$HYPERML_DATA_3" ] && mkdir -p $HYPERML_DATA_3
      [ ! -f "$HYPERML_DATA_3/HyperTritonTree_18q.root"  ] && alien_cp alien:/alice/cern.ch/user/p/pfecchio/Trees/HyperTritonTree_18q.root  $HYPERML_DATA_3/HyperTritonTree_18q.root
      [ ! -f "$HYPERML_DATA_3/HyperTritonTree_18r.root"  ] && alien_cp alien:/alice/cern.ch/user/p/pfecchio/Trees/HyperTritonTree_18r.root  $HYPERML_DATA_3/HyperTritonTree_18r.root
      [ ! -f "$HYPERML_DATA_3/HyperTritonTree_19d2.root" ] && alien_cp alien:/alice/cern.ch/user/p/pfecchio/Trees/HyperTritonTree_19d2.root $HYPERML_DATA_3/HyperTritonTree_19d2.root

      [ ! -d "$HYPERML_TABLES_3" ] && mkdir -p $HYPERML_TABLES_3
fi