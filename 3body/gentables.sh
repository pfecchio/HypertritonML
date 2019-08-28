#!/bin/bash
FORCE=0
DATA=0
MC=0

for arg in "$@"
do
    if [ "$arg" == "--force" ] || [ "$arg" == "-f" ]; then
        FORCE=1
    fi

    if [ "$arg" == "data" ]; then
        DATA=1
    fi

    if [ "$arg" == "mc" ]; then
        MC=1
    fi

    if [ "$arg" == "all" ]; then
        MC=1
        DATA=1
    fi
done

if [ $MC -eq 1 ]; then
    if [ $FORCE -eq 1 ]; then
        rm "$HYPERML_TABLES_3/HyperTritonTable_19d2.root"
    fi
    [ ! -f "$HYPERML_TABLES_3/HyperTritonTable_19d2.root" ] && root -l -q -b TreesToTables/GenerateTableFromMC.cc || echo "HyperTritonTable_19d2.root already exist! Will not be generated again."
fi

if [ $DATA -eq 1 ]; then
    if [ $FORCE -eq 1 ]; then
        rm "$HYPERML_TABLES_3/HyperTritonTable_18q.root"
    fi
    [ ! -f "$HYPERML_TABLES_3/HyperTritonTable_18q.root" ] && root -l -q -b TreesToTables/GenerateTableFromData.cc || echo "HyperTritonTable_18q.root already exist! Will not be generated again."
fi
