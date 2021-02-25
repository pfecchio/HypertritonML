#!/bin/bash

# require ROOT
command -v root >/dev/null 2>&1 || { echo >&2 "ROOT required but it's not available.  Aborting."; return; }

N_BODY=0

# manage script arguments
usage() { echo "Usage: $0 [-b|--body <2|3>] [-h|--help]" 1>&2; }

[ $# -eq 0 ] && usage && return

while [[ $# -gt 0 ]]
do
key="$1"

      case $key in
            -b | --body) # Specify decay channel, 2 or 3.
                  N_BODY=$2
                  (($2 == 2 || $2 == 3)) || { usage; return;}
                  shift; shift
                  ;;
            -h | --help) # Display help
                  usage
                  shift
                  return
                  ;;
      esac
done

HYPERML_CODE="$PWD"
HYPERML_COMMON="$HYPERML_CODE/common"
HYPERML_RESULTS="$HYPERML_CODE/Results"

# code python importable
export PYTHONPATH="${PYTHONPATH}:$HYPERML_COMMON/TrainingAndTesting:$HYPERML_COMMON/Utils"

export HYPERML_TREES_${N_BODY}="$HYPERML_CODE/Trees/${N_BODY}Body"
export HYPERML_TABLES_${N_BODY}="$HYPERML_CODE/Tables/${N_BODY}Body"
export HYPERML_FIGURES_${N_BODY}="$HYPERML_CODE/Figures/${N_BODY}Body"
export HYPERML_MODELS_${N_BODY}="$HYPERML_CODE/Models/${N_BODY}Body"
export HYPERML_UTILS_${N_BODY}="$HYPERML_CODE/Utils/${N_BODY}Body"
export HYPERML_RESULTS_${N_BODY}="$HYPERML_CODE/Results/${N_BODY}Body"
export HYPERML_EFFICIENCIES_${N_BODY}="$HYPERML_RESULTS/${N_BODY}Body/Efficiencies"

export HYPERML_UTILS="$HYPERML_CODE/Utils"

# BlastWaveFits.root is required
[ ! -f $HYPERML_UTILS/BlastWaveFits.root ] && echo "$HYPERML_UTILS/BlastWaveFits.root not found!" 
