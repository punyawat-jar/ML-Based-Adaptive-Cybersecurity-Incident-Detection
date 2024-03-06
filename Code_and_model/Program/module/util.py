import pandas as pd
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

def progress_bar(*args, **kwargs):
    bar = tqdm(*args, **kwargs)

    def checker(x):
        bar.update(1)
        return False

    return checker

def changeLabel(df, label):
    df.loc[df['label'].isin(["normal", "BENIGN"]), "label"] = 0
    df.loc[df['label'] == label, "label"] = 1
    
    rows_to_drop = df.loc[(df['label'] != 0) & (df['label'] != 1)].index
    
    df = df.drop(rows_to_drop)
    
    df['label'] = df['label'].astype('int')
    df = scaler(df)

    return df

def scaler(df):
    scaler = MinMaxScaler()
    for col in df.columns:
        if col != 'label':
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


def process_data(Data, train_index, test_index, window_size):
    def rearrange_sequences(generator, index, window_size):
        rearranged_data = []

        for idx in tqdm(index, desc="Processing sequences"):
            if idx >= window_size - 1:
                sequence_end = idx + 1
                sequence_start = sequence_end - window_size 
                
                batch_x = generator.data[sequence_start:sequence_end]
                batch_y = generator.targets[idx] 

                rearranged_data.append((batch_x, batch_y))

        return rearranged_data
    # Process training data
    print('Training data processing...')
    train_data = rearrange_sequences(Data, train_index, window_size)
    gc.collect()
    
    print('Testing data processing...')
    test_data = rearrange_sequences(Data, test_index, window_size)
    del rearrange_sequences
    gc.collect()
    return train_data, test_data
    
def separate_features_labels(data, window_size):
    features = np.array([item[0] for item in data], dtype=np.float32)
    labels = np.array([item[1] for item in data], dtype=np.float32) 
    features = features.reshape(-1, window_size, features.shape[-1])

    return features, labels
