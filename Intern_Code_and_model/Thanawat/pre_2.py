# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 17/01/2020
"""
Pre-processing program to extract window-related normalized entropy from Netflow files

Parameters
----------
window_width  : window width in seconds
window_stride : window stride in seconds
data          : pandas DataFrame of the Netflow file

Return
----------
Create 3 output files:
- data_window3_botnetx.h5         : DataFrame with the extracted data: Sport (RU), DstAddr (RU), Dport (RU)
- data_window_botnetx_id3.npy     : Numpy array containing SrcAddr
- data_window_botnetx_labels3.npy : Numpy array containing Label
"""

import pandas as pd
import numpy as np
import datetime
import h5py

from scipy.stats import mode

window_width = 120 # seconds
window_stride = 60 # seconds

print("Import data")
data = pd.read_csv("/home/s2316001/merge.csv")
#with pd.option_context('display.max_rows', None, 'display.max_columns', 15):
#    print(data.shape)
#    print(data.head())
#    print(data.dtypes)

print("Preprocessing")
def normalize_column(dt, column):
    mean = dt[column].mean()
    std = dt[column].std()
    print(mean, std)

    dt[column] = (dt[column]-mean) / std

data['StartTime'] = pd.to_datetime(data['StartTime']).astype(np.int64)*1e-9
datetime_start = data['StartTime'].min()

data['Window_lower'] = (data['StartTime']-datetime_start-window_width)/window_stride+1
data['Window_lower'].clip(lower=0, inplace=True)
data['Window_upper_excl'] = (data['StartTime']-datetime_start)/window_stride+1
data = data.astype({"Window_lower": int, "Window_upper_excl": int})
data.drop('StartTime', axis=1, inplace=True)

data['Label'], labels = pd.factorize(data['Label'].str.slice(0, 15))
#print(data.dtypes)

def RU(df):
    if df.shape[0] == 1:
        return 1.0
    else:
        proba = df.value_counts()/df.shape[0]
        h = proba*np.log10(proba)
        return -h.sum()/np.log10(df.shape[0])

X = pd.DataFrame()
nb_windows = data['Window_upper_excl'].max()
print(nb_windows)

for i in range(0, nb_windows):
    gb = data.loc[(data['Window_lower'] <= i) & (data['Window_upper_excl'] > i)].groupby('SrcAddr')
    X = X.append(gb.agg({'Sport':[RU], 
                         'DstAddr':[RU], 
                         'Dport':[RU]}).reset_index())
    print(X.shape)

del(data)

X.columns = ["_".join(x) if isinstance(x, tuple) else x for x in X.columns.ravel()]
#print(X.columns.values)

#print(X.columns.values)
columns_to_normalize = list(X.columns.values)
columns_to_normalize.remove('SrcAddr_')

normalize_column(X, columns_to_normalize)

with pd.option_context('display.max_rows', 10, 'display.max_columns', 22):
    print(X.shape)
    print(X)
    print(X.dtypes)
    
#with pd.option_context('display.max_rows', 10, 'display.max_columns', 20):
#    print(X.loc[X['Label'] != 0])

X.drop('SrcAddr_', axis=1).to_hdf('/home/s2316001/codepaper/data_window3.h5', key="data", mode="w")
np.save("/home/s2316001/codepaper/data_window_id3.npy", X['SrcAddr_'])
np.save("/home/s2316001/codepaper/data_window_labels3.npy", labels)