import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def progress_bar(*args, **kwargs):
    from tqdm import tqdm
    bar = tqdm(*args, **kwargs)

    def checker(x):
        bar.update(1)
        return False

    return checker

def changeLabel(df, label):
    df.loc[df['label'].isin(["normal", "BENIGN"]), "label"] = 0
    df.loc[df['label'] == label, "label"] = 1
    df.loc[(df['label'] != 0) & (df['label'] != 1), "label"] = 0
    
    df['label'] = df['label'].astype('int')
    df = scaler(df)

    return df

def scaler(df):
    scaler = MinMaxScaler()
    for col in df.columns:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).ravel()
    return df


def check_data_template(data_template):
    if 'kdd' in data_template.lower():
        data_template = 'kdd'
    
    elif 'cic' in data_template.lower():
        data_template = 'cic'

    else:
        ## In cases the it is not the default dataset (NSL-KDD, CIC-IDS2017). Please implements the data_template after this line.
        raise Exception('Please enter the default dataset or implements the training dataset besed on your dataset')
    return data_template