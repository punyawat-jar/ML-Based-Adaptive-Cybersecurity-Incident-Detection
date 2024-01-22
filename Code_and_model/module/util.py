from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def progress_bar(*args, **kwargs):
    bar = tqdm(*args, **kwargs)

    def checker(x):
        bar.update(1)
        return False

    return checker

def changeLabel(df, label):
    print(label)
    df.loc[df['label'] == "normal", "label"] = 0
    df.loc[df['label'] == label, "label"] = 1
    df.loc[(df['label'] != 0) & (df['label'] != 1), "label"] = 0
    
    df['label'] = df['label'].astype('int')
    return df

def label_preprocess(df, label):
    df = changeLabel(df, label)
    scaler = MinMaxScaler()
    for col in df.columns:
        if col != 'label':  # Skip the 'label' column
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).ravel()
    
    return df