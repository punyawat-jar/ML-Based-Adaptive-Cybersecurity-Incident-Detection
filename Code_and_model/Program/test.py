import os
import pandas as pd
import numpy as np
import argparse
import glob
import joblib
import concurrent.futures
import time
import traceback
import sys
import json

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from module.file_op import *
from module.util import progress_bar, check_data_template, scaler
from module.testing_module import *

def main():
    
    parser = argparse.ArgumentParser(description='Testing code')
    parser.add_argument('--data',
                        dest='data_template',
                        type=str,
                        required=True,
                        help='The data struture. The default data structures is cic (CICIDS2017) and kdd (NSL-KDD). (*Require)')

    parser.add_argument('--chooese_model',
                        dest='chooese_csv',
                        type=str,
                        help="The csv file's location for input an integration's models. The csv file must contains attack column which match with model column.")
    
    parser.add_argument('--model',
                        dest='model_loc',
                        type=str,
                        help='The trained models loaction.')
    
    parser.add_argument('--network',
                        dest='net_file_loc',
                        type=str,
                        help='The netowrk file location (.csv)')

    parser.add_argument('--sequence_mode',
                        dest='sequence',
                        action='store_true',
                        help='The sequence mode show the network in sequence with the prediction of each attack')
    
    parser.add_argument('--debug',
                        dest='debug',
                        type=bool,
                        default=True,
                        help='The debug model show the list of the prediction categories that might be an attack of the model.')

    arg = parser.parse_args()

    data_template = arg.data_template

    chooese_csv = arg.chooese_csv if arg.chooese_csv is not None else f'./{data_template}/model.csv'

    model_loc =  arg.model_loc if arg.model_loc is not None else f'{data_template}/Training/model'

    net_file_loc = arg.net_file_loc

    sequence_mode = arg.sequence
    debug_mode = arg.debug
    
    try:
        #Parameter & path setting
        models_loc = []
        os.chdir('./Code_and_model/Program') ##Change Working Directory
        
        result_path = f'{data_template}/Result'
        weight_path = f'{data_template}/weight.json'
        
        check_file(chooese_csv)
        
        data_template = check_data_template(data_template)
        df = getDataset(arg, data_template, net_file_loc)

        
        df_train = pd.read_csv(glob.glob(f'{data_template}/train_test_folder/train_{data_template}/*')[0], skiprows=progress_bar())
        df_test = pd.read_csv(glob.glob(f'{data_template}/train_test_folder/test_{data_template}/*')[0], skiprows=progress_bar())
        
        # train_index, test_index = check_train_test_index(df_train, df_test)
        
        
        X_train = df_train.drop(['label', 'Unnamed: 0'], axis=1)
        X_test = df_test.drop(['label', 'Unnamed: 0'], axis=1)
        
        y_train = df_train['label']
        y_test = df_test['label']
        
        processAttack(y_test)
        
        y_test = y_test.values.astype(int)
        
        #Reading Weight from file, if exist. if not calculated from the dataset (Default)
        CheckWegihtFileCreated = creating_weight_file(weight_path)
        
        if CheckWegihtFileCreated == False:
            filtered_labels = [label for label in y_train if label not in ('normal', 'BENIGN')]
            label_counts = Counter(filtered_labels)
            total_labels = len(filtered_labels)
            label_percentages = {label: (count / total_labels) * 100 for label, count in label_counts.items()}
            
            with open(weight_path, "w") as jsonfile: 
                json.dump(label_percentages, jsonfile)
        else:
            with open(weight_path) as jsonfile:
                label_percentages = json.load(jsonfile)
                
        lowest_percent_attack = min(label_percentages, key=label_percentages.get)
        threshold = label_percentages[lowest_percent_attack]

        model_df = pd.read_csv(chooese_csv)[['attack', 'model']]
        model_df = model_df[model_df['attack'] != 'normal.csv']
        
        files = glob.glob(model_loc+'/**', recursive=True)

        for file in files:
            for _, row in model_df.iterrows():
                if row['attack'] in file and row['model'] in file:
                    print(f'model file = {file}')
                    models_loc.append(file)
                    break

        models = read_model(models_loc, model_df, label_percentages)

        print(f'-- Evaluation the model with {len(models)} attacks--')
        y_pred, time_pred, attack_df = prediction(models, sequence_mode, threshold, X_test)
        print(y_test)
        print(y_pred)
        y_pred_df = pd.DataFrame(y_pred, columns=['Prediction'])
        

        evalu = evaluation(y_test, y_pred, data_template, result_path)
        
        result_df = pd.concat([X_test, attack_df], axis=1)
        
        y_pred_df.index = result_df.index
        result_df = pd.concat([result_df, y_pred_df], axis=1)
        result_df.to_csv(f'{result_path}/attack_prediction_{data_template}.csv', index = True)
        
        print(f'Accuracy : {evalu["accuracy"]}\nF1-score : {evalu["f1"]}\nPrecision : {evalu["precision"]}\nRecall : {evalu["recall"]}')
    except Exception as E:
        print(E)
        traceback.print_exc()
        sys.exit(1)
if __name__ == '__main__':
    main()