import numpy as np
from multiprocessing import Pool, cpu_count
import pandas as pd 
import sys
import traceback
from sklearn.preprocessing import MinMaxScaler
from module.util import *
from module.file_op import *
from tqdm import tqdm
def read_train(path):
    feature=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
          "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
          "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
          "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
          "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
          "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

    dftrain = pd.read_csv(path, names = feature)
    return dftrain

def read_test(path):
    feature=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
          "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
          "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
          "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
          "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
          "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

    dftest = pd.read_csv(path, names = feature)
    return dftest

def align_columns(dataset1, dataset2, column_sequence):
    # Reindex both datasets with the aligned column order
    dataset1 = dataset1.reindex(columns=column_sequence)
    dataset2 = dataset2.reindex(columns=column_sequence)

    # Set NaN values to 0
    dataset1 = dataset1.fillna(0)
    dataset2 = dataset2.fillna(0)

    return dataset1, dataset2

def check_column_differences(dataset1, dataset2):
    columns1 = set(dataset1.columns)
    columns2 = set(dataset2.columns)

    columns_only_in_dataset1 = columns1 - columns2
    columns_only_in_dataset2 = columns2 - columns1

    return columns_only_in_dataset1, columns_only_in_dataset2



def column_manage(df):
    df.drop(['difficulty'],axis=1,inplace=True)
    df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'], dtype='int')
    return df


def process_label(args):
    label, _, mix_directory, df_template, _ = args  
    if label == 'normal':
        print(f'Skip {label}')
        return None
    
    df_temp = df_template.copy()
    
    df_temp = changeLabel(df_temp, label)

    
    for col in df_temp.columns:
        if df_temp[col].dtype == 'bool':
            df_temp[col] = df_temp[col].astype(int)
    
   
    df_temp.to_csv(f"./kdd/{mix_directory}/{label}.csv", index=False)
    

def ProcessKDD(file_path, input_dataset, multiCPU, num_processes=cpu_count()):
    try:
        mix_directory = './dataset/mix_dataset'
        if input_dataset is None:
            data = []
            
            Same_fileName, file_type = checkFileName(file_path)
            
            if Same_fileName != True:          ## Check if file name different
                raise Exception('All the file must be in the same type')
            
            if len(file_path) != 2 and input_dataset is None:             ## Check if only train and test dataset
                raise Exception('Please make sure there only train+, test+ dataset, if you improvise your own dataset please preprocess and load ')
            elif input_dataset is not None:
                df = pd.read_csv(input_dataset)
            else:
                train_files = list_of_file_contain('train', file_path)
                test_files = list_of_file_contain('test', file_path)
                
                if file_type == 'txt':             ## .txt file must be Train+.txt
                    data.append(read_train(train_files[0]))
                    data.append(read_test(test_files[0]))
                    
                    data[0], data[1] = align_columns(data[0], data[1], data[0].columns)
                    
                elif file_type == 'csv':
                    data[0] = pd.read_csv(train_files)
                    data[1] = pd.read_csv(test_files)
                    
                df = pd.concat(data, ignore_index=True)
                df = column_manage(df)
                df = scaler(df)
                df.to_csv('./kdd/KDD.csv', index=False)
                print(f'Shape of dataset : {df.shape}')
        
        else:
            df = pd.read_csv(input_dataset)
    
        labels = df.label.value_counts().index.tolist()
        
        if multiCPU:
            print(f'Using Multiprocessing with : {num_processes}')
            df_template = df.copy()

            args_list = [(label, index, mix_directory, df_template, len(labels)) for index, label in enumerate(labels)]

            with Pool(processes=num_processes) as pool:
                list(tqdm(pool.imap_unordered(process_label, args_list), total=len(args_list)))

            print('Preprocessing KDD Done')
        
        else:
            print('Using single CPU')
            for i, label in tqdm(enumerate(labels)):
                if label == 'normal':
                    print(f'Skip {label}')
                else:
                    df_temp = df.copy()
                    print(f'Starting {label} {i+1}/{len(labels)}')
                    
                    df_temp = changeLabel(df_temp, label)

                    for col in df.columns:
                        if df[col].dtype == 'bool':
                            df[col] = df[col].astype(int)
                            
                    df_temp.to_csv(f"./kdd/{mix_directory}/{label}.csv", index=False)
            print('Preprocessing KDD Done')
        
        
    except Exception as E:
        print(E)
        traceback.print_exc()
        sys.exit(1)