import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import joblib
import h5py
for z in range (1,14):
    df = pd.DataFrame()
    for k in range (1,14):
        if k != z:
            print(f'---------read{k}------------')
            new_data = pd.read_csv(f'/home/s2316001/Clean_data/train{k}.csv')
            df = pd.concat([df,new_data])
    X_en = df.drop(columns=[df.columns[0],"Label"], axis=1)
    y_en = df['Label']
    model = RandomForestClassifier(n_estimators=100)  # You can adjust the hyperparameters as needed
    model.fit(X_en, y_en)
    filename = f'/home/s2316001/model_rfe/new12/rfe_model12to1no{z}.sav'
    joblib.dump(model, filename)
    print(f'-------done {z}----------')