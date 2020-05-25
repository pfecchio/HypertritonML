#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np

import pandas as pd
import seaborn as sns

CENT_CLASSES = [(0, 90)]
PT_BINS = [2, 10]
CT_BINS = list(range(0, 35))


MC_PATH = Path(os.environ['HYPERML_TABLES_3'] + '/O2/signal_reweighted_df.parquet.gzip')

def main():
    df_mc = pd.read_parquet(MC_PATH)
    print(df_mc.head())

    for cclass in CENT_CLASSES:
        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
            for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                # data_range = f'{ctbin[0]}<ct<{ctbin[1]} and {ptbin[0]}<pt<{ptbin[1]} and {cclass[0]}<=centrality<{cclass[1]}'
                # pt selection should be restored when possible
                data_range = f'{ctbin[0]}<ct<{ctbin[1]} and {cclass[0]}<=fCent<{cclass[1]}'

                df_tmp = df_mc.query(data_range)

                print(len(df_mc))
                print (len(df_tmp))
                
                del df_tmp



if __name__ == "__main__":
    main()
