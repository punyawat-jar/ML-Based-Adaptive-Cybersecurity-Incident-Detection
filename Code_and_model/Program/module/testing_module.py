
import pandas as pd
import numpy as np

import joblib
import concurrent.futures
import time


import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
import json


from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from keras.models import load_model
from module.file_op import *
from module.util import progress_bar, scaler
from module.model import sequential_models

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
                
                if file.endswith('.joblib'):
                    model = joblib.load(file)
                    weight = weight_data.get(attack_type, 0)
                    model_object = ModelObject(attack_type, model_name, model, weight)
                    models.append(model_object)
                    break
                
                elif file.endswith('.h5'):
                    model = load_model(file)
                    weight = weight_data.get(attack_type, 0)
                    model_object = ModelObject(attack_type, model_name, model, weight)
                    models.append(model_object)
                    break

    return models

def processAttack(df):
    print('processAttack Data...')
    # np_array[(np_array == 'normal') | (np_array == 'BENIGN')] = 0
    # np_array[np_array != 0] = 1
    df.loc[df['label'].isin(['normal', 'BENIGN']), 'label'] = 0
    # Replace all other values with 1
    df.loc[df['label'] != 0, 'label'] = 1
    return df

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

        # attacks_MoreThanOne(attack_df)

        print(f'The predictioin shape :{predict.shape}')
        print(f'prediction time in mean : {np.mean(pre_time)} seconds')
        return predict, pre_time, attack_df

    else:
        print('== The model opeartion in prediction overall mode')
        print('Predicting...')
        pred, time, attack_df = predictionModel(models, X_test, threshold)

        # print(attack_df)

        # attacks_MoreThanOne(attack_df)

        print(f'The prediction : {pred} using {time:0.4f} seconds')
        return pred, time, attack_df

    
# def attacks_MoreThanOne(attack_df):
#     mask = attack_df['attack'].apply(lambda x: len(x) > 1)

#     # Filter the DataFrame based on the mask
#     filtered_df = attack_df[mask]

#     # print(filtered_df)

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

    
def process_Largest_remainder_method(label_percentages, decimal, weight_path):
    label_percentages_rounded = {key: round(value, decimal) for key, value in label_percentages.items()}
    label_percentages_fractional = {key: ((value - int(value))*(10**decimal)) - int((value - int(value))*(10**decimal)) for key, value in label_percentages.items()}
    
    ishundred = round(100 - sum(label_percentages_rounded.values()),decimal)
    if ishundred != 0:
        if ishundred < 0:   ## if the sum is over 100, will -- 0.001 each. From least weight of weight cut to most.
            sort_Ascending = dict(sorted(label_percentages_fractional.items(), key=lambda item: (item[1], item[0])))
            for label, _ in sort_Ascending.items():
                label_percentages_rounded[label] -= (10**-decimal)
                ishundred += (10**-decimal)

                if ishundred == 0:
                    break
        else:               ## if the sum is less than 100, will ++ 0.001 each. From most weight of weight cut to least.
            sort_Decending = dict(sorted(label_percentages_fractional.items(), key=lambda item: (-item[1], item[0])))
            
            for label, _ in sort_Decending.items():
                label_percentages_rounded[label] += (10**-decimal)
                ishundred -= (10**-decimal)

                if ishundred == 0:
                    break
                
    with open(weight_path, "w") as jsonfile:
        json.dump(label_percentages_rounded, jsonfile)

    return label_percentages_rounded

def classification_evaluation(test_labels, attack_labels):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    # Iterate through all the labels and predictions
    for true_label, predictions in zip(test_labels, attack_labels):
        if isinstance(predictions, set):
            predictions = list(predictions)  # Convert set to list if necessary
        
        if not predictions:  # Treat empty predictions as 'normal'
            predictions = ['normal']

        if true_label in predictions:
            true_positives += 1  # True label is among the predictions
            false_positives += len(predictions) - 1  # Other predictions are considered false positives
        else:
            false_negatives += 1  # True label was not predicted
            false_positives += len(predictions)  # All predictions are considered false positives

        # Assuming 'normal' is the negative class
        if 'normal' in predictions and true_label == 'normal':
            true_negatives += 1  # Correctly predicted normal

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    return accuracy, f1, precision, recall