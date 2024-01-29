import os
import traceback
import argparse
import sys
import glob
import gc
import shutil

import joblib
from joblib import dump

from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from module.model import getModel
from module.util import progress_bar, check_data_template
from module.file_op import *
from module.discord import *

class joblib_model:
    def __init__(self, model, weight):
        self.model = joblib.load(model)
        self.weight = weight


def best_model_for_attack(model_folder):
    bestmodel = {'attack': [], 'model': [], 'accuracy': [], 'f1': []}
    for model in model_folder:
        modelname = model.split('_')[-1]
        df = pd.read_csv(model).sort_values(['f1', 'accuracy'], ascending=[False, False])

        # Check if the DataFrame is empty
        if df.empty:
            # Handle the empty DataFrame case (e.g., append None or default values)
            bestmodel['attack'].append(modelname)
            bestmodel['model'].append(None)
            bestmodel['accuracy'].append(None)
            bestmodel['f1'].append(None)
        else:
            bestmodel['attack'].append(modelname)
            bestmodel['model'].append(df.iloc[0].iloc[0])
            bestmodel['accuracy'].append(df.iloc[0]['accuracy'])
            bestmodel['f1'].append(df.iloc[0]['f1'])

    return pd.DataFrame(data=bestmodel)
    



# This train.py file will train each model separately

def main():
    try:
        parser = argparse.ArgumentParser(description='Training code')
        parser.add_argument('--data',
                    dest='data_template',
                    type=str,
                    required=True,
                    help='The data struture. The default data structures is cic (CICIDS2017) and kdd (NSL-KDD). (*Require)')

        arg = parser.parse_args()
        
        data_template = arg.data_template
        
        #File path
        os.chdir('./Code_and_model') ##Change Working Directory
    
        models = getModel()
        
        dataset_paths = glob.glob('.\\mix_dataset\\*.csv')
        
        f1_scores = []
        precision_scores = []
        recall_scores = []
        cms = []
        FPRs = []
        FNRs = []
        TPRs = []
        TNRs = []

        check_data_template(data_template)
        
        if data_template == 'cic':
            full_data = './cic/CICIDS2017.csv'
        elif data_template == 'kdd':
            full_data = './kdd/KDD.csv'
        else:
            raise Exception('The dataset template is not regcognize (cic or kdd)'
                            )
        main_df = pd.read_csv(full_data, low_memory=False, skiprows=progress_bar(), stratify=y)

        X_main = main_df.drop('label', axis=1)
        y_main = main_df['label']

        # Split the main dataset
        X_train_main, X_test_main, _, _ = train_test_split(X_main, y_main, test_size=0.3, random_state=42, stratify=y_main)

        # Get the indices of the training and testing sets
        train_index = X_train_main.index
        test_index = X_test_main.index

        saveTrain_test = './/train_test_folder'
        train_test_folder = [f'.{data_template}/train_test_folder/train_{data_template}',
                            f'.{data_template}/train_test_folder/test_{data_template}']
    
        
        for dataset_path in tqdm(dataset_paths, desc="Dataset paths"):
            # Load and preprocess dataset
            print(f'== reading {dataset_path} ==')
            df = pd.read_csv(dataset_path, low_memory=False, skiprows=progress_bar())
            # Splitting data
            X = df.drop('label', axis=1)
            y = df['label']

            sub_X_train = X.loc[train_index]
            sub_y_train = y.loc[train_index]
            sub_X_test = X.loc[test_index]
            sub_y_test = y.loc[test_index]
            
            # Concatenate X_train with y_train, and X_test with y_test
            train_combined = pd.concat([sub_X_train, sub_y_train], axis=1)
            test_combined = pd.concat([sub_X_test, sub_y_test], axis=1)

            backslash = '\\'
            
            del df
            del X
            del y
            gc.collect()
            # Train and evaluate models on the current dataset
            results = {}
            dataset_name = dataset_path.split(backslash)[-1]  # Name of the dataset

            #To be debugged, deleted if the same
            #======
            train_combined.to_csv(f'.//{train_test_folder[0]}//train_{dataset_name}.csv', index=False)
            test_combined.to_csv(f'.//{train_test_folder[1]}//test_{dataset_name}.csv', index=False)
            #======

            for name, model in tqdm(models.items(), desc=f"Training {data_template} Models"):

                send_discord_message(f'== Mix {data_template} Training: {dataset_name} with model: {name} ==')
                print(f'== Mix {data_template} Training: {dataset_name} with model: {name} ==')
                
                model.fit(sub_X_train, sub_y_train)
                y_pred = model.predict(sub_X_test)

                accuracy = accuracy_score(sub_y_test, y_pred)
                f1 = f1_score(sub_y_test, y_pred)
                precision = precision_score(sub_y_test, y_pred)
                recall = recall_score(sub_y_test, y_pred)
                conf_matrix = confusion_matrix(sub_y_test, y_pred, labels=model.classes_)

                # Extract metrics from confusion matrix
                TN, FP, FN, TP = conf_matrix.ravel()
                
                conf_matrix_path = f'{data_template}/Training/confusion_martix/{dataset_name}'

                if not os.path.exists(conf_matrix_path):
                    os.makedirs(conf_matrix_path)
                    
                # Plot and save confusion matrix
                # Your existing code for confusion matrix
                cm_dis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
                fig, ax = plt.subplots()
                cm_dis.plot(ax=ax)
                fig.savefig(f'{data_template}/Training/confusion_martix/{dataset_name}/{dataset_name}_{name}_confusion_matrix.png')

                plt.close(fig)
                
                # Here I assume binary classification for loss (0 or 1). Adjust if needed.
                loss = np.mean(np.abs(y_pred - sub_y_test))
                send_discord_message(f'== {data_template} Done Training: {dataset_name} with model: {name}, acc: {accuracy}, loss: {loss}, f1: {f1} ==')
                print(f'== Done Training: {dataset_name} with model: {name}, acc: {accuracy}, loss: {loss}, f1: {f1} ==')
                
                models_save_path = f'{data_template}/Training/model/{dataset_name}'
                
                
                if not os.path.exists(models_save_path):
                    os.makedirs(models_save_path)

                # Save the trained model
                model_filename =  f".{data_template}/Training/model/{dataset_name}/{dataset_name}_{name}_model.joblib"
                dump(model, model_filename)
                print(f"== {data_template} Model {name} saved as {model_filename} ==")
                
                results[name] = [accuracy, loss, f1, precision, recall, conf_matrix]


                # Convert results to DataFrame and save with dataset name
                # shutil.move(f'{dataset_path}', f'./dataset\\\\mix_done\\{dataset_name}')
                result_df = pd.DataFrame.from_dict(results, orient='index', columns=['accuracy', 'loss', 'f1', 'precision', 'recall', 'confusion_matrix'])
                result_filename = f".{data_template}/Training/compare/evaluation_results_{dataset_name}"
                result_df.to_csv(result_filename)
                gc.collect()
            # send_discord_message('== @everyone All training and evaluation is done ==')
            print('== All training and evaluation is done ==')

        #Assemble the results
        compare_data = glob.glob(f'./{data_template}/Training/compare/*.csv')
        compare_df = best_model_for_attack(compare_data)
        compare_df.to_csv(f'.{data_template}/model.csv')
        
    except Exception as E:
        print(E)
        traceback.print_exc()
        sys.exit(1)



if __name__ == '__main__':
    main()