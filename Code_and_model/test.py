import os
import pandas as pd
import numpy as np
import argparse
import glob
import joblib

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
    
def read_model(models_loc, df):
    models_dict = {}

    for file in models_loc:
        for i, row in df.iterrows():
            if '.' in row['attack']:
                row['attack'] = row['attack'].split('.')[0]
            if row['attack'] in file and row['model'] in file:
                models_dict[row['attack']] = {row['model']: joblib.load(file)}
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

    sequence_model = arg.sequence

    debug_model = arg.debug
    
    models_loc = []
    os.chdir('./Code_and_model')
    
    check_file(chooese_csv)
    print('-- Reading Dataset --')
    model_df = pd.read_csv(chooese_csv)[['attack', 'model']]
    model_df = model_df[model_df['attack'] != 'normal.csv']
    
    files = glob.glob(model_loc+'/*.csv/**', recursive=True)

    for file in files:
        for i, row in model_df.iterrows():
            if row['attack'] in file and row['model'] in file:
                models_loc.append(file)
                break

    models = read_model(models_loc, model_df)
    


    for attack in models:
        #Handling the key if more than 1 (Let's expert choose)
        model_key = list(models[attack].keys())
        if len(model_key) != 1:
            print('Please choose the model by number :')
        else:
            model_key = model_key[0]
        
        model = models[attack][model_key]

        print(f"Attack: {attack}, Model_name: {model_key}, model: {model}")

        model.prediction()

if __name__ == '__main__':
    main()