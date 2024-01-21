import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
import h5py
for i in range (0,50):
    X_train = pd.DataFrame()
    X_val= pd.DataFrame()
    y_train= pd.DataFrame()
    y_val= pd.DataFrame()
    print(f'----------- Round {i}------------------')
    for j in range (1,14):
        for z in range (1,14):
            if z!=j:
                new_data = pd.read_csv(f"/home/s2316001/Clean_data/Clean_{z}-XX.csv")
                print(new_data.shape)
                X = new_data.drop(columns=[new_data.columns[0],"Label"], axis=1)
                y = new_data['Label']
                X_train_temp, X_val_temp, y_train_temp, y_val_temp = train_test_split(X, y, test_size=0.3)
                df_train_each = pd.concat([X_train_temp,y_train_temp],axis=1)
                df_test_each = pd.concat([X_val_temp,y_val_temp],axis=1)
                df_train_each.rename(columns = {0:'Label'}, inplace = True)
                df_test_each.rename(columns = {0:'Label'}, inplace = True)
                df_train_each.to_csv(f'/home/s2316001/Clean_data/Merge13_split/file{z}/train{i}_No{z}.csv', encoding='utf-8')
                df_test_each.to_csv(f'/home/s2316001/Clean_data/Merge13_split/file{z}/test{i}_No{z}.csv', encoding='utf-8')
                X_train = pd.concat([X_train,X_train_temp])
                X_val = pd.concat([X_val,X_val_temp])
                y_train = pd.concat([y_train,y_train_temp])
                y_val = pd.concat([y_val,y_val_temp])
        df_train = pd.concat([X_train,y_train],axis=1)
        df_test = pd.concat([X_val,y_val],axis=1)
        df_train.rename(columns = {0:'Label'}, inplace = True)
        df_test.rename(columns = {0:'Label'}, inplace = True)
        df_train.to_csv(f'/home/s2316001/Clean_data/Merge13_split/train{i}_No{z}.csv', encoding='utf-8')
        df_test.to_csv(f'/home/s2316001/Clean_data/Merge13_split/test{i}_No{z}.csv', encoding='utf-8')