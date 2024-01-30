import numpy as np
import glob
import pandas as pd 
import gc


from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys
import traceback

from module.file_converter import _to_utf8
from module.file_op import *
from module.util import *
from module.discord import send_discord_message

def renaming_class_label(df: pd.DataFrame):
    def labels(label):
        if 'Web Attack' in label:
            if 'Brute Force' in label:
                return 'Web Attack-Brute Force'
            elif 'XSS' in label:
                return 'Web Attack-XSS'
            elif 'Sql Injection' in label:
                return 'Web Attack-Sql Injection'
        return label

    df['label'] = df['label'].apply(labels)
    
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

    _to_utf8(file_paths) if all(keyword in file_paths for keyword in ['Thursday', 'Morning']) else None
    print('Reading CIC-IDS2017 data...')
    df_list = [pd.read_csv(file, low_memory=False, skiprows=progress_bar()) for file in file_paths]
    
    df = pd.concat(df_list, ignore_index=True)
    
    del df_list
    gc.collect()

    nRow, nCol = df.shape
    print(f'The CIC-IDS2017 dataset has {nRow} rows and {nCol} columns')
    
    return df

def create_df_single_process(df, labels, directory):
    for i, label in enumerate(labels):
        if label == 'BENIGN':
            print(f'Skip {label}')
            return

        df_temp = df.copy()
        print(f'Starting {label} {i+1}/{len(labels)}')
        df_temp = changeLabel(df_temp, label)
        df_temp.to_csv(f"./cic/{directory}/{label}.csv", index=False)


def create_df_multiprocess(args):
    label, index, total_labels, df_template, directory = args
    
    if label == 'BENIGN':
        print(f'Skip {label}')
        return
    
    df_temp = df_template.copy()
    # print(f'Starting {label} {index+1}/{total_labels}')
    df_temp = changeLabel(df_temp, label)
    output_path = os.path.join("./cic", directory, f"{label}.csv")
    df_temp.to_csv(output_path, index=False)

def ProcessCIC(df_loc, input_dataset, multiCPU, num_processes=cpu_count()):
    try:
        directory = './dataset/mix_dataset'
        if input_dataset is None:
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
            df = pd.concat([df.drop('destination_port_priority', axis=1), temp], axis=1)
            
            del temp
            gc.collect()
            
            #Drop the inf and NaN values
            print('Dropping inf and Nan values...')
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            cols_with_nan = df.columns[df.isna().any()].tolist()
            print("Columns containing NaN values:", cols_with_nan)
            
            #Change the attack name
            renaming_class_label(df)
            
            #Sort the dataset by time
            df = df.sort_values(by='timestamp')
            
            non_numerical_columns = df.select_dtypes(exclude=['number']).columns
            print(f'The column to drop :{non_numerical_columns}, but except "label"')
            df = df.drop(non_numerical_columns.drop(['label']), axis = 1)
            print(df.shape)
            print('Saving CIC_IDS2017.csv ...')
            df = scaler(df)
            df.to_csv('./cic/CIC_IDS2017.csv', index=False)
            
        else:
            print('Reading Dataset from user input...')
            df = pd.read_csv(f'{input_dataset}', skiprows=progress_bar())
            
            
        
        labels = df.label.value_counts().index.tolist()
        print('Preprocessing Dataset...')


        if multiCPU:
            print(f'Using Multiprocessing with : {num_processes}')
            args_list = [(label, index, len(labels), df, directory) for index, label in enumerate(labels)]
            with Pool(processes=num_processes) as pool:
                for _ in tqdm(pool.imap_unordered(create_df_multiprocess, args_list), total=len(args_list)):
                    pass 
        else:
            print('Using single CPU')
            create_df_single_process(df, labels, directory)
        print('Preprocessing Done')
    except Exception as E:
        print(E)
        traceback.print_exc()
        sys.exit(1)