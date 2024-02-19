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

    parser.add_argument('--sequence_mode',
                        dest='sequence',
                        action='store_true',
                        help='The sequence mode show the network in sequence with the prediction of each attack')
    
    arg = parser.parse_args()

    data_template = arg.data_template

    chooese_csv = f'./{data_template}/model.csv'

    model_loc =  f'{data_template}/Training/model'

    # net_file_loc = arg.net_file_loc

    sequence_mode = arg.sequence
    # debug_mode = arg.debug
    
    try:
        #Parameter & path setting
        models_loc = []
        weight_decimal = 3 # decimal position (e.g. 0.001)
        
        result_path = f'{data_template}/Result'
        weight_path = f'{data_template}/weight.json'
        
        check_file(chooese_csv)
        
        data_template = check_data_template(data_template)
        # df = getDataset(arg, data_template, net_file_loc)

        
        df_train = pd.read_csv(glob.glob(f'{data_template}/train_test_folder/train_{data_template}/*')[0], skiprows=progress_bar())
        df_test = pd.read_csv(glob.glob(f'{data_template}/train_test_folder/test_{data_template}/*')[0], skiprows=progress_bar())
        
        test_labels = df_test['label'].copy()
        df_test = processAttack(df_test)
        
        # X_train = df_train.drop(['label', 'Unnamed: 0'], axis=1)
        X_test = df_test.drop(['label', 'Unnamed: 0'], axis=1)
        
        y_train = df_train['label']
        y_test = df_test['label']
        
        y_test = y_test.values.astype(int)
        
        #Reading Weight from file, if exist. if not calculated from the dataset (Default)
        CheckWegihtFileCreated = creating_weight_file(weight_path)
        
        og_attack_percent = read_attack_percent(y_train, weight_decimal)
        
        if CheckWegihtFileCreated == True:
            print('Loading weight files')
            with open(weight_path) as jsonfile:
                label_percentages = json.load(jsonfile)
        else:
            label_percentages = og_attack_percent
            if weight_path is not None:
                with open(weight_path, "w") as jsonfile:
                    json.dump(label_percentages, jsonfile)
        lowest_percent_attack = min(og_attack_percent, key=og_attack_percent.get)
        threshold = og_attack_percent[lowest_percent_attack]
        print(f'threshold = {threshold}')
        model_df = pd.read_csv(chooese_csv)[['attack', 'model']]
        model_df = model_df[model_df['attack'] != 'normal.csv']
        
        files = glob.glob(model_loc+'/**', recursive=True)

        for file in files:
            for _, row in model_df.iterrows():
                if row['attack'] in file and row['model'] in file:
                    print(f'Reading model:  {file}')
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
        attack_labels = result_df['attack']
        
    
        precision, recall, f1_score, accuracy = classification_evaluation(test_labels, attack_labels)
        print('Evaluation by binary classification')
        print(f'Accuracy : {evalu["accuracy"]}\nF1-score : {evalu["f1"]}\nPrecision : {evalu["precision"]}\nRecall : {evalu["recall"]}')
        
        print('Evaluation by multiclass classification')
        print(f'Accuracy : {accuracy}\nF1-score : {f1_score}\nPrecision : {precision}\nRecall : {recall}')
        
        print('Default attack percentage:')
        print(f'test label : {test_labels}')
        print(read_attack_percent(test_labels, weight_decimal))
        
        print('Adaptive tuning attack percentage:')
        y_detect_bi = y_test & y_pred
        
        y_detect_mul = calculate_adaptive(test_labels, y_detect_bi, data_template)
        print(read_attack_percent(y_detect_mul, weight_decimal))
        
        
        
        
    except Exception as E:
        print(E)
        traceback.print_exc()
        sys.exit(1)
if __name__ == '__main__':
    main()