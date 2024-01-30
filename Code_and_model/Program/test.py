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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from module.file_op import *
from module.util import progress_bar, check_data_template, scaler

class ModelObject:
    def __init__(self, attack_type, model_name, model, weight):
        self.attack_type = attack_type
        self.model_name = model_name
        self.model = model
        self.weight = weight
    
def read_model(models_loc, df, weight_data):
    print('-- Reading Trained Model --')
    models = []

    for file in tqdm(models_loc):
        for _, row in df.iterrows():
            if '.' in row['attack']:
                attack_type = row['attack'].split('.')[0]
            else:
                attack_type = row['attack']

            if attack_type in file and row['model'] in file:
                model_name = row['model']
                # print(file)
                model = joblib.load(file)
                weight = weight_data.get(attack_type, 0)
                model_object = ModelObject(attack_type, model_name, model, weight)
                models.append(model_object)
                break

    return models

def processAttack(np_array):
    print('processAttack Data...')
    np_array[(np_array == 'normal') | (np_array == 'BENIGN')] = 0
    np_array[np_array != 0] = 1
    return np_array

def printModel(models):
    for model in models:
        #Handling the key if more than 1 (Let's expert choose)
        print(f"Attack: {model.attack_type}, Model_name: {model.model_name}, model: {model.model} with weight :{model.weight}")

def predictionModel(models, X_test, threshold):
    # Function to make predictions with a single model
    futures = []

    def make_prediction(model, X_test):
        y_pred = model.model.predict(X_test)
        
        return y_pred, model.weight, model.attack_type

    
    start_all = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for model in models:
            future = executor.submit(make_prediction, model, X_test)
            futures.append(future)

    attacks_per_row = np.empty((len(X_test), 0)).tolist()

    for future in futures:
        y_pred, _, attack_type = future.result()
        # Vectorize the appending process
        indices = np.where(y_pred == 1)[0]
        for i in indices:
            attacks_per_row[i].append(attack_type)


    predictions = [future.result() for future in futures]

    attack_df = pd.DataFrame({'attack': attacks_per_row})
    
    final_prediction = integrate_predictions(predictions, threshold)
    end_all = time.perf_counter()
    
    prediction_time = end_all - start_all

    return final_prediction, prediction_time, attack_df

def integrate_predictions(predictions, threshold):
    # linear integration: w1x1 + w2x2 + ...
    weighted_sum = np.sum([pred * weight for pred, weight, _ in predictions], axis=0)

    # Normalize by total weight
    total_weight = sum(weight for _, weight, _ in predictions)
    
    final_pred = weighted_sum / total_weight

    # Threshold condition to each element
    final_pred = np.where(final_pred >= threshold, 1, 0)
    
    return final_pred

def prediction(models, sequence_mode, threshold, X_test):
    if sequence_mode == True:           #Sequence mode is to check the prediction for each element.
        print('== The model operation in sequence mode ==')
        predict = []
        pre_time = []
        attacks = pd.DataFrame()

        for i in tqdm(range(0,X_test.shape[0])):
            pred, time, attack_df = predictionModel(models, X_test[i:i+1], threshold)
            predict.extend(pred)
            pre_time.append(time)
            attacks = pd.concat([attacks, attack_df], ignore_index=True)
            print(f'The prediction is : {pred} using {time:0.4f} seconds')

        predict = np.array(predict)
        print(attacks)

        attacks_MoreThanOne(attack_df)

        print(f'The predictioin shape :{predict.shape}')
        print(f'prediction time in mean : {np.mean(pre_time)} seconds')
        return predict, pre_time, attack_df
    
    else:
        print('== The model opeartion in prediction overall mode')
        print('Predicting...')
        pred, time, attack_df = predictionModel(models, X_test, threshold)

        # print(attack_df)

        attacks_MoreThanOne(attack_df)

        print(f'The prediction : {pred} using {time:0.4f} seconds')
        return pred, time, attack_df
    
def attacks_MoreThanOne(attack_df):
    mask = attack_df['attack'].apply(lambda x: len(x) > 1)

    # Filter the DataFrame based on the mask
    filtered_df = attack_df[mask]

    # print(filtered_df)

def checkShape(y_test, y_pred):
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()

    print('----- Checking y label -----')
    print("Shape of y_test:", y_test.shape)
    print("Shape of y_pred:", y_pred.shape)

    print("Data type of elements in y_test:", y_test.dtype)
    print("Data type of elements in y_pred:", y_pred.dtype)

    print('-----------------------------')

def evaluation(y_test, y_pred, data_template, result_path):
    evalu = {}
    checkShape(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    evalu['accuracy'] = acc
    evalu['f1'] = f1
    evalu['precision'] = precision
    evalu['recall'] = recall
    evalu['conf_matrix'] = [conf_matrix.ravel()]

    pd.DataFrame.from_dict(evalu).to_csv(f"./{result_path}/result.csv", index=False)


    makeConfusion(conf_matrix, data_template)
    return evalu


def makeConfusion(conf_matrix, data_template):

    plt.figure()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'./confusion_matrix_{data_template}.png')
    plt.close()

def getDataset(arg, data_template, net_file_loc):
    if net_file_loc is not None:
        net_file_loc = arg.net_file_loc
    elif data_template == 'cic':
        net_file_loc = './cic/CIC_IDS2017.csv'
    elif data_template == 'kdd':
        net_file_loc = './kdd/KDD.csv'
    else:
        raise Exception('The Dataset is not found.')
    
    print('-- Reading Dataset --')
    df = pd.read_csv(net_file_loc, skiprows=progress_bar())
    df = scaler(df)
    print('-- Reading Dataset successfully --')
    
    return df

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
            label_counts = Counter(y_train)
            total_labels = len(y_train)
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