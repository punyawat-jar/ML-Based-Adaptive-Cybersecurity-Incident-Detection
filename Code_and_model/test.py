import os
import pandas as pd
import numpy as np
import argparse
import glob
import joblib
import concurrent.futures
import time


import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# from sklearn.preprocessing import MinMaxScaler

# from sklearn.linear_model import LogisticRegression, Perceptron
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB

class ModelObject:
    def __init__(self, attack_type, model_name, model, weight):
        self.attack_type = attack_type
        self.model_name = model_name
        self.model = model
        self.weight = weight

def preprocess(np_array):
    print('-- Preprocessing Data --')
    np_array[(np_array == 'normal') | (np_array == 'BENIGN')] = 0
    np_array[np_array != 0] = 1
    return np_array

def makePath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_file(path):
    if os.path.isfile(path):
        print(f'{path} exist')
    else:
        raise Exception(f'Error: {path} not exist')
    
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
                print(file)
                model = joblib.load(file)
                weight = weight_data.get(attack_type, 0)

                model_object = ModelObject(attack_type, model_name, model, weight)
                models.append(model_object)
                break

    return models

def printModel(models):
    for model in models:
        #Handling the key if more than 1 (Let's expert choose)
        print(f"Attack: {model.attack_type}, Model_name: {model.model_name}, model: {model.model} with weight :{model.weight}")

def predictionModel(models, X_test, threshold):
    # Function to make predictions with a single model
    
    def make_prediction(model, X_test):

        y_pred = model.model.predict(X_test)
        # print(f'-- Evaluation for {model.model_name} for {model.attack_type}. The prediction is {y_pred} --')
        return y_pred, model.weight

    futures = []
    start_all = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:

        for model in models:
            future = executor.submit(make_prediction, model, X_test)
            futures.append(future)

    predictions = [future.result() for future in futures]
    
    # Integrate predictions
    final_prediction = integrate_predictions(predictions, threshold)
    end_all = time.perf_counter()
    prediction_time = end_all - start_all
    return final_prediction, prediction_time

def integrate_predictions(predictions, threshold):
    # linear integration: w1x1 + w2x2 + ...
    weighted_sum = np.sum([pred * weight for pred, weight in predictions], axis=0)

    # Normalize by total weight
    total_weight = sum(weight for _, weight in predictions)
    final_pred = weighted_sum / total_weight

    # Threshold condition to each element
    final_pred = np.where(final_pred >= threshold, 1, 0)
    
    return final_pred

def prediction(models, sequence_mode, threshold, X_test):
    if sequence_mode == True:           #Sequence mode is to check the prediction for each element.
        print('== The model operation in sequence mode ==')
        predict = []
        pre_time = []
        for i in tqdm(range(1,X_test.shape[0])):
        # for i in tqdm(range(1,20)):      #Debug
            pred, time = predictionModel(models, X_test[i-1:i], threshold)
            predict.extend(pred)
            pre_time.append(time)
            print(f'The prediction is : {pred} using {time:0.4f} seconds')

        predict = np.array(predict)
        print(f'The predictioin shape :{predict.shape}')
        print(f'prediction time in mean : {np.mean(pre_time)} seconds')
        return predict, pre_time
    
    else:
        print('== The model opeartion in prediction overall mode')
        print('Predicting...')
        pred, time = predictionModel(models, X_test, threshold)
        print(f'The prediction : {pred} using {time:0.4f} seconds')
        return pred, time
    
def makeConfusion(conf_matrix, data_template):
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'.\\confusion_matrix_{data_template}.png')
    plt.close()

def progress_bar(*args, **kwargs):
    bar = tqdm(*args, **kwargs)

    def checker(x):
        bar.update(1)
        return False

    return checker

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
                        required=True,
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

    model_loc =  arg.model_loc if arg.model_loc is not None else f'./{data_template}/model'

    net_file_loc = arg.net_file_loc

    sequence_mode = arg.sequence
    print(sequence_mode)
    debug_mode = arg.debug
    
    models_loc = []
    os.chdir('./Code_and_model')
    
    check_file(chooese_csv)
    check_file(net_file_loc)
    print('-- Reading Dataset --')

    # df = pd.concat([chunk for chunk in tqdm(pd.read_csv(net_file_loc, chunksize=1000), desc='Loading dataset')])
    df = pd.read_csv(net_file_loc, skiprows=progress_bar())
    print('-- Reading Dataset successfully --')
    X = df.drop('label', axis=1)
    y = df['label']
    _, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    preprocess(y_test)
    y_test = y_test.values
    y_test = y_test.astype(int)

    label_counts = Counter(y_train)
    total_labels = len(y_train)
    label_percentages = {label: (count / total_labels) * 100 for label, count in label_counts.items()}
    # print(label_percentages)

    lowest_percent_attack = min(label_percentages, key=label_percentages.get)
    threshold = label_percentages[lowest_percent_attack]

    model_df = pd.read_csv(chooese_csv)[['attack', 'model']]
    model_df = model_df[model_df['attack'] != 'normal.csv']
    
    files = glob.glob(model_loc+'/*.csv/**', recursive=True)

    for file in files:
        for _, row in model_df.iterrows():
            if row['attack'] in file and row['model'] in file:
                models_loc.append(file)
                break

    models = read_model(models_loc, model_df, label_percentages)
    # print(models)

    # printModel(models)
    print(f'-- Evaluation the model with {len(models)} attacks--')
    y_pred, time_pred = prediction(models, sequence_mode, threshold, X_test)
    
    print("Shape of y_test:", y_test.shape)
    print("Shape of y_pred:", y_pred.shape)

    

    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()

    print('----- Checking y label -----')
    print("after Shape of y_test:", y_test.shape)
    print("after Shape of y_pred:", y_pred.shape)

    print("Data type of elements in y_test:", y_test.dtype)
    print("Data type of elements in y_pred:", y_pred.dtype)

    print('-----------------------------')
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Extract metrics from confusion matrix
    TN, FP, FN, TP = conf_matrix.ravel()
    
    conf_matrix_path = f'.\\{data_template}'
    makePath(conf_matrix_path)
    print(f'Accuracy : {acc}\nF1-score : {f1}\nPrecision : {precision}\nRecall : {recall}')

if __name__ == '__main__':
    main()