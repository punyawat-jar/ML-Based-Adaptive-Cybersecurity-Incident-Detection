import numpy as np
import requests, json
import os
import pandas as pd 
import multiprocessing
import gc
from sklearn.preprocessing import MinMaxScaler
os.chdir("C:\\Users\\Kotani Lab\\Desktop\\ML_senior_project\\ML-Based-Adaptive-Cybersecurity-Incident-Detection\\Code_and_model\\cic\\dataset")

def preprocess(df):
    df.drop(['timestamp'], axis =1, inplace=True)
    # df.loc[df['label'] == "BENIGN", "label"] = 0
    # df.loc[df['label'] != 0, "label"] = 1
    
    replacements = {
    'Web Attack ÃÂÃÂ Brute Force': 'Web Attack-Brute Force',
    'Web Attack ÃÂÃÂ XSS': 'Web Attack-XSS',
    'Web Attack ÃÂÃÂ Sql Injection': 'Web Attack-Sql Injection'
    }
    
    df['label'] = df['label'].replace(replacements)
    scaler = MinMaxScaler()
    for col in df.columns:
        if col != 'label':  # Skip the 'label' column
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).ravel()
    return df
    
print('Reading Dataset')
df = pd.read_csv('Original_dataset-notuse\\CIC_IDS2017.csv')
print('Preprocessing Dataset')
df = preprocess(df.copy())
print('Saving Dataset')
df.to_csv("CIC_real_with_char_attack.csv", index=False)
print('Done')