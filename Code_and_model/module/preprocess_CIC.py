import numpy as np
import glob
import pandas as pd 
import gc
from sklearn.preprocessing import MinMaxScaler

from module.file_converter import _to_utf8
from module.util import progress_bar

def renaming_class_label(df: pd.DataFrame):
    labels = {
    'Web Attack ÃÂÃÂ Brute Force': 'Web Attack-Brute Force',
    'Web Attack ÃÂÃÂ XSS': 'Web Attack-XSS',
    'Web Attack ÃÂÃÂ Sql Injection': 'Web Attack-Sql Injection'
    }

    df['label'] = df['label'].replace(labels)
    
    return df

def categorize_port(port):
    if 0 <= port <= 1023:
        return 1
    elif 1024 <= port <= 49151:
        return 2
    elif 49152 <= port <= 65535:
        return 3
    else:
        return None  # or however you want to handle out-of-range values

def concatFiles(df_loc):
    #Concat Files
    file_paths = [file for file in df_loc if file.endswith('.csv')]
    
    for file in file_paths:
        if file.find('thursday') and file.fild('morning'):
            _to_utf8(file)
    
    df_list = [pd.read_csv(file) for file in file_paths]
    df = pd.concat(df_list, ignore_index=True)

    del df_list
    gc.collect()

    nRow, nCol = df.shape
    print(f'The CIC-IDS2017 dataset has {nRow} rows and {nCol} columns')
    
    return df

def preprocess(df):
    scaler = MinMaxScaler()
    for col in df.columns:
        if col != 'label':  # Skip the 'label' column
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).ravel()
    return df
    

def ProcessCIC(df_loc):
    col_list = []
    #Concat the files in dataset
    df = concatFiles(df_loc)
    
    #Prepare the dataset for training
    for _, name in enumerate(df.columns):
        if name[0] == ' ':
            name = name[1:]
        if ' ' in name:
            name = name.replace(' ', '_')
        col_list.append(name)
    df.columns = col_list
    
    
    #Change the port to 3 categories 
    df['Destination_Port'] = df['Destination_Port'].apply(categorize_port)
    
    list = df.columns.tolist()
    list = [x.lower() for x in list]
    df.columns = list
    df = df.rename(columns = {'destination_port': 'destination_port_priority'})
    
    
    #One-hot encode the 'destination_port_priority' column
    temp = pd.get_dummies(df['destination_port_priority'], prefix='destination_port_priority', dtype=int)
    df = df.drop('destination_port_priority', axis=1).join(temp)
    del temp
    gc.collect()
    
    #Drop the inf and NaN values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    cols_with_nan = df.columns[df.isna().any()].tolist()
    print("Columns containing NaN values:", cols_with_nan)
    
    #Change the attack name
    renaming_class_label(df)
    
    #Sort the dataset by time
    df = df.sort_values(by='timestamp')
    
    df = df.drop(['source_port', 'timestamp'], axis = 1)
    
    df.to_csv('CIC_IDS2017.csv', index=False, skiprows=progress_bar())
    
    print('Preprocessing Dataset')
    df = preprocess(df.copy())
    
    