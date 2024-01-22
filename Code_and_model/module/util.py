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
    df.loc[df['label'].isin(["normal", "BENIGN"]), "label"] = 0
    df.loc[df['label'] == label, "label"] = 1
    df.loc[(df['label'] != 0) & (df['label'] != 1), "label"] = 0
    
    df['label'] = df['label'].astype('int')

    return df

