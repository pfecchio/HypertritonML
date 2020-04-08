#!/bin/bash

[ -z "$ALICE_PHYSICS" ] && echo "AliPhysics environment required! Load it before sourcing this." && return

HYPERML_DATA="$PWD/Trees"
HYPERML_TABLES="$PWD/Tables"
HYPERML_FIGURES="$PWD/Figures"
HYPERML_RESULTS="$PWD/Results"
HYPERML_MODELS="$PWD/Models"
HYPERML_CODE="$PWD"
HYPERML_COMMON="$HYPERML_CODE/common"

export PYTHONPATH="${PYTHONPATH}:$HYPERML_COMMON/TrainingAndTesting:$HYPERML_COMMON/Utils"
export HYPERML_UTILS="$PWD/Utils"

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

[ ! -d $HYPERML_UTILS ] && mkdir -p $HYPERML_UTILS
[ ! -f $HYPERML_UTILS/BlastWaveFits.root ] && alien_cp /alice/cern.ch/user/m/mpuccio/hyper_data/BlastWaveFits.root file://$HYPERML_UTILS/BlastWaveFits.root

if [ $BODY_2 -eq 1 ]; then    
      export HYPERML_DATA_2="$HYPERML_DATA/2Body"
      export HYPERML_TABLES_2="$HYPERML_TABLES/2Body"
      export HYPERML_FIGURES_2="$HYPERML_FIGURES/2Body"
      export HYPERML_MODELS_2="$HYPERML_MODELS/2Body"
      export HYPERML_UTILS_2="$HYPERML_UTILS/2Body"
      export HYPERML_RESULTS_2="$HYPERML_RESULTS/2Body"
      export HYPERML_EFFICIENCIES_2="$HYPERML_RESULTS_2/Efficiencies"

      [ ! -d "$HYPERML_DATA_2" ] && mkdir -p $HYPERML_DATA_2
      [ ! -d "$HYPERML_TABLES_2" ] && mkdir -p $HYPERML_TABLES_2
      [ ! -d "$HYPERML_FIGURES_2" ] && mkdir -p $HYPERML_FIGURES_2
      [ ! -d "$HYPERML_MODELS_2" ] && mkdir -p $HYPERML_MODELS_2
      [ ! -d "$HYPERML_EFFICIENCIES_2" ] && mkdir -p $HYPERML_EFFICIENCIES_2
      [ ! -d "$HYPERML_RESULTS_2" ] && mkdir -p $HYPERML_RESULTS_2
      [ ! -d "$HYPERML_UTILS_2" ] && mkdir -p $HYPERML_UTILS_2
      [ ! -f "$HYPERML_DATA_2/HyperTritonTree_18q.root"  ] && alien_cp /alice/cern.ch/user/f/fmazzasc/tree/HyperTritonTree_18q.root  file://$HYPERML_DATA_2/HyperTritonTree_18q.root
      [ ! -f "$HYPERML_DATA_2/HyperTritonTree_18r.root"  ] && alien_cp /alice/cern.ch/user/f/fmazzasc/tree/HyperTritonTree_18r.root  file://$HYPERML_DATA_2/HyperTritonTree_18r.root
      [ ! -f "$HYPERML_DATA_2/HyperTritonTree_19d2.root" ] && alien_cp /alice/cern.ch/user/f/fmazzasc/tree/HyperTritonTree_19d2.root file://$HYPERML_DATA_2/HyperTritonTree_19d2.root
      [ ! -f "$HYPERML_DATA_2/HyperTritonTree_18qLS.root" ] && alien_cp /alice/cern.ch/user/a/alitrain/PWGLF/LF_PbPb/1108_20190920-2344/merge/HyperTritonTree.root file://$HYPERML_DATA_2/HyperTritonTree_18qLS.root
      [ ! -f "$HYPERML_DATA_2/HyperTritonTree_18rLS.root" ] && alien_cp /alice/cern.ch/user/a/alitrain/PWGLF/LF_PbPb/1109_20190920-2345/merge/HyperTritonTree.root file://$HYPERML_DATA_2/HyperTritonTree_18rLS.root
      [ ! -f "$HYPERML_UTILS/He3TPCCalibration.root" ] && alien_cp /alice/cern.ch/user/m/mpuccio/hyper_data/He3TPCCalibration.root file://$HYPERML_UTILS/He3TPCCalibration.root
      [ ! -f "$HYPERML_UTILS/absorption.root" ] && alien_cp /alice/cern.ch/user/m/mpuccio/hyper_data/absorption.root file://$HYPERML_UTILS/absorption.root
fi

if [ $BODY_3 -eq 1 ]; then    
      export HYPERML_DATA_3="$HYPERML_DATA/3Body"
      export HYPERML_TABLES_3="$HYPERML_TABLES/3Body"
      export HYPERML_FIGURES_3="$HYPERML_FIGURES/3Body"
      export HYPERML_MODELS_3="$HYPERML_MODELS/3Body"
      export HYPERML_EFFICIENCIES_3="$HYPERML_EFFICIENCIES/3Body"
      export HYPERML_UTILS_3="$HYPERML_UTILS/3Body"
      export HYPERML_RESULTS_3="$HYPERML_RESULTS/3Body"
      export HYPERML_EFFICIENCIES_3="$HYPERML_RESULTS_3/Efficiencies"

      [ ! -d "$HYPERML_DATA_3" ] && mkdir -p $HYPERML_DATA_3
      [ ! -d "$HYPERML_TABLES_3" ] && mkdir -p $HYPERML_TABLES_3
      [ ! -d "$HYPERML_FIGURES_3" ] && mkdir -p $HYPERML_FIGURES_3
      [ ! -d "$HYPERML_MODELS_3" ] && mkdir -p $HYPERML_MODELS_3
      [ ! -d "$HYPERML_RESULTS_3" ] && mkdir -p $HYPERML_RESULTS_3
      [ ! -d "$HYPERML_EFFICIENCIES_3" ] && mkdir -p $HYPERML_EFFICIENCIES_3
      [ ! -d "$HYPERML_UTILS_3" ] && mkdir -p $HYPERML_UTILS_3      
      [ ! -f "$HYPERML_DATA_3/HyperTritonTreeBkg_18q.root"  ] && alien_cp /alice/cern.ch/user/a/alitrain/PWGLF/LF_PbPb/1218_20200122-1031/merge/HyperTritonTree.root  file://$HYPERML_DATA_3/HyperTritonTreeBkg_18q.root
      [ ! -f "$HYPERML_DATA_3/HyperTritonTreeBkg_18r.root"  ] && alien_cp /alice/cern.ch/user/a/alitrain/PWGLF/LF_PbPb/1219_20200122-1031/merge/HyperTritonTree.root  file://$HYPERML_DATA_3/HyperTritonTreeBkg_18r.root
      # [ ! -f "$HYPERML_DATA_3/HyperTritonTreeData_18q.root"  ] && alien_cp /alice/cern.ch/user/a/alitrain/PWGLF/LF_PbPb/1218_20200122-1031/merge/HyperTritonTree.root  file://$HYPERML_DATA_3/HyperTritonTreeData_18q.root
      # [ ! -f "$HYPERML_DATA_3/HyperTritonTreeData_18r.root"  ] && alien_cp /alice/cern.ch/user/a/alitrain/PWGLF/LF_PbPb/1219_20200122-1031/merge/HyperTritonTree.root  file://$HYPERML_DATA_3/HyperTritonTreeData_18r.root
      # [ ! -f "$HYPERML_DATA_3/AnalysisResults_18q.root"  ] && alien_cp /alice/cern.ch/user/a/alitrain/PWGLF/LF_PbPb/1218_20200122-1031/merge/HyperTritonTree.root  file://$HYPERML_DATA_3/AnalysisResults_18q.root
      # [ ! -f "$HYPERML_DATA_3/AnalysisResults_18r.root"  ] && alien_cp /alice/cern.ch/user/a/alitrain/PWGLF/LF_PbPb/1219_20200122-1031/merge/HyperTritonTree.root  file://$HYPERML_DATA_3/AnalysisResults_18r.root
      [ ! -f "$HYPERML_DATA_3/HyperTritonTree_19d2a.root" ] && alien_cp /alice/cern.ch/user/a/alitrain/PWGLF/LF_PbPb_MC/1494_20200331-2313_child_1/merge/HyperTritonTree.root file://$HYPERML_DATA_3/HyperTritonTree_19d2a.root
      [ ! -f "$HYPERML_DATA_3/HyperTritonTree_19d2b.root" ] && alien_cp /alice/cern.ch/user/a/alitrain/PWGLF/LF_PbPb_MC/1494_20200331-2313_child_2/merge/HyperTritonTree.root file://$HYPERML_DATA_3/HyperTritonTree_19d2b.root
      [ ! -f "$HYPERML_DATA_3/HyperTritonTree_19d2c.root" ] && alien_cp /alice/cern.ch/user/a/alitrain/PWGLF/LF_PbPb_MC/1494_20200331-2313_child_3/merge/HyperTritonTree.root file://$HYPERML_DATA_3/HyperTritonTree_19d2c.root
fi
