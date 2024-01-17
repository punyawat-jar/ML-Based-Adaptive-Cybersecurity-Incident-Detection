import os
import pandas as pd
import numpy as np
import argparse
import glob
import joblib

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def check_file(path):
    if os.path.isfile(path):
        print(f'{path} exist')
    else:
        raise Exception(f'Error: {path} not exist')
    
def read_model(models_loc, df, weight_data):
    models_dict = {}

    for file in models_loc:
        for i, row in df.iterrows():
            if '.' in row['attack']:
                attack_type = row['attack'].split('.')[0]
            else:
                attack_type = row['attack']
            
            if attack_type in file and row['model'] in file:
                model_name = row['model']
                model = joblib.load(file)
                weight = weight_data.get(attack_type, 0)

                if attack_type not in models_dict:
                    models_dict[attack_type] = {}
                models_dict[attack_type][model_name] = model
                models_dict[attack_type]['weight'] = weight
                break
    return models_dict



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

    parser.add_argument('--sequence',
                        dest='sequence',
                        type=bool,
                        default=False,
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

    sequence_model = arg.sequence
    
    debug_model = arg.debug
    
    models_loc = []
    os.chdir('./Code_and_model')
    
    check_file(chooese_csv)
    check_file(net_file_loc)
    print('-- Reading Dataset --')

    df = pd.read_csv(net_file_loc)
    
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    label_counts = Counter(y_train)
    total_labels = len(y_train)
    label_percentages = {label: (count / total_labels) * 100 for label, count in label_counts.items()}
    print(label_percentages)
    lowest_percent_attack = min(label_percentages, key=label_percentages.get)
    treshold = label_percentages[lowest_percent_attack]

    model_df = pd.read_csv(chooese_csv)[['attack', 'model']]
    model_df = model_df[model_df['attack'] != 'normal.csv']
    
    files = glob.glob(model_loc+'/*.csv/**', recursive=True)

    for file in files:
        for i, row in model_df.iterrows():
            if row['attack'] in file and row['model'] in file:
                models_loc.append(file)
                break

    models = read_model(models_loc, model_df, label_percentages)
    # print(models)


    for attack in models:
        #Handling the key if more than 1 (Let's expert choose)
        model_key = list(models[attack].keys())
        model_key = model_key[0]
        # print(model_key)

        model = models[attack][model_key]
        weight = models[attack]['weight']
        print(f"Attack: {attack}, Model_name: {model_key}, model: {model} with weight :{models[attack]['weight']}")

    # for attack in models:
    #     model_key = list(models[attack].keys())
    #     model = models[attack][model_key]
    #     print('-- Starting Evaluation --')
    #     y_pred = model.predict(X_test)

        

if __name__ == '__main__':
    main()