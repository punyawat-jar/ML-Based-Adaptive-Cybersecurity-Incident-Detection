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
from multiprocessing import Pool, cpu_count

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from module.model import getModel
from module.util import progress_bar, check_data_template, scaler
from module.file_op import *
from module.discord import *
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
class joblib_model:
    def __init__(self, model, weight):
        self.model = joblib.load(model)
        self.weight = weight


def best_model_for_attack(model_folder):
    bestmodel = {
        'attack': [],
        'model': [],
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': []
    }
    
    for model in model_folder:
        modelname = model.split('_')[-1].split('.')[0]
        df = pd.read_csv(model).sort_values(['f1', 'accuracy'], ascending=[False, False])

        # Check if the DataFrame is empty
        if df.empty:
            bestmodel['attack'].append(modelname)
            bestmodel['model'].append(None)
            bestmodel['accuracy'].append(None)
            bestmodel['f1'].append(None)
            bestmodel['precision'].append(None)
            bestmodel['recall'].append(None)
        else:
            bestmodel['attack'].append(modelname)
            bestmodel['model'].append(df.iloc[0].iloc[0])
            bestmodel['accuracy'].append(df.iloc[0]['accuracy'])
            bestmodel['f1'].append(df.iloc[0]['f1'])
            bestmodel['precision'].append(df.iloc[0]['precision'])
            bestmodel['recall'].append(df.iloc[0]['recall'])

    return pd.DataFrame(data=bestmodel)
    

def train_and_evaluate_Multiprocess(args):
    try:
        name, model, data_template, dataset_name, sub_X_train, sub_y_train, sub_X_test, sub_y_test = args
        
        # send_discord_message(f'== Mix {data_template} Training: {dataset_name} with model: {name} ==')
        # print(f'== Mix {data_template} Training: {dataset_name} with model: {name} ==')
        
        model.fit(sub_X_train, sub_y_train)
        y_pred = model.predict(sub_X_test)

        accuracy = accuracy_score(sub_y_test, y_pred)
        f1 = f1_score(sub_y_test, y_pred, zero_division=0)
        precision = precision_score(sub_y_test, y_pred, zero_division=0)
        recall = recall_score(sub_y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(sub_y_test, y_pred, labels=model.classes_)
        
        conf_matrix_path = f'{data_template}/Training/confusion_matrix/{dataset_name}'
        if not os.path.exists(conf_matrix_path):
            os.makedirs(conf_matrix_path)
            
        cm_dis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
        fig, ax = plt.subplots()
        cm_dis.plot(ax=ax)
        fig.savefig(f'{conf_matrix_path}/{dataset_name}_{name}_confusion_matrix.png')
        plt.close(fig)
        
        loss = np.mean(np.abs(y_pred - sub_y_test))
        # send_discord_message(f'== {data_template} Done Training: {dataset_name} with model: {name}, acc: {accuracy}, loss: {loss}, f1: {f1} ==')
        # print(f'== Done Training: {dataset_name} with model: {name}, acc: {accuracy}, loss: {loss}, f1: {f1} ==')
        
        models_save_path = f'{data_template}/Training/model/{dataset_name}'
        
        if not os.path.exists(models_save_path):
            os.makedirs(models_save_path)

        model_filename = f"{models_save_path}/{dataset_name}_{name}_model.joblib"
        dump(model, model_filename)
        # print(f"== {data_template} Model {name} saved as {model_filename} ==")
    
    except ValueError as ve:
        if "covariance is ill defined" in str(ve):
            print("Skipping due to ill-defined covariance for dataset:", dataset_name)
            # Consider logging this error or handling it in a way that marks this dataset/model as skipped
            return None  # or return some indication of failure/skipping for this task
        else:
            raise
    
    return {name: [accuracy, loss, f1, precision, recall, conf_matrix]}

def train_and_evaluation_singleprocess(models, data_template, data, dataset_name, results):
    for name, model in models.items():
        sub_X_train, sub_X_test, sub_y_train, sub_y_test = data
        
        models_save_path = f'{data_template}/Training/model/{dataset_name}'
        conf_matrix_path = f'{data_template}/Training/confusion_martix/{dataset_name}'
        makePath(models_save_path)
        makePath(conf_matrix_path)
        
        # send_discord_message(f'== Mix {data_template} Training: {dataset_name} with model: {name} ==')
        # print(f'== Mix {data_template} Training: {dataset_name} with model: {name} ==')
        
        model.fit(sub_X_train, sub_y_train)
        y_pred = model.predict(sub_X_test)

        accuracy = accuracy_score(sub_y_test, y_pred)
        f1 = f1_score(sub_y_test, y_pred, zero_division=0)
        precision = precision_score(sub_y_test, y_pred, zero_division=0)
        recall = recall_score(sub_y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(sub_y_test, y_pred)


        cm_dis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        fig, ax = plt.subplots()
        cm_dis.plot(ax=ax)
        fig.savefig(f'{data_template}/Training/confusion_martix/{dataset_name}/{dataset_name}_{name}_confusion_matrix.png')

        plt.close(fig)
        
        loss = np.mean(np.abs(y_pred - sub_y_test))
        # send_discord_message(f'== {data_template} Done Training: {dataset_name} with model: {name}, acc: {accuracy}, loss: {loss}, f1: {f1} ==')
        # print(f'== Done Training: {dataset_name} with model: {name}, acc: {accuracy}, loss: {loss}, f1: {f1} ==')
        
        

        model_filename =  f"{data_template}/Training/model/{dataset_name}/{dataset_name}_{name}_model.joblib"
        dump(model, model_filename) 
        # print(f"== {data_template} Model {name} saved as {model_filename} ==")
        
        results[name] = [accuracy, loss, f1, precision, recall, conf_matrix]
        
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['accuracy', 'loss', 'f1', 'precision', 'recall', 'confusion_matrix'])
        result_filename = f"{data_template}/Training/compare/evaluation_results_{dataset_name}"
        result_df.to_csv(result_filename)
        gc.collect()
        
        
# This train.py file will train each model separately

def main():
    try:
        parser = argparse.ArgumentParser(description='Training code')
        parser.add_argument('--data',
                    dest='data_template',
                    type=str,
                    required=True,
                    help='The data struture. The default data structures is cic (CICIDS2017) and kdd (NSL-KDD). (*Require)')
        
        parser.add_argument('--multiProcess',
                            dest='multiCPU',
                            action=argparse.BooleanOptionalAction,
                            help='multiCPU is for using all the process.')
        
        parser.add_argument('--n_Process',
                            dest='num_processes',
                            type=str,
                            help='num_processes is the number of process by user, default setting is all process (cpu_count()).')
        
        arg = parser.parse_args()
        
        num_processes = int(arg.num_processes) if arg.num_processes is not None else cpu_count()
        
        data_template = arg.data_template
        multiCPU = arg.multiCPU
        
        #File path
        os.chdir('./Code_and_model/Program') ##Change Working Directory
    
        models = getModel()
        
        dataset_paths = glob.glob(f'{data_template}/dataset/mix_dataset/*.csv')
        
        check_data_template(data_template)
        
        if data_template == 'cic':
            full_data = './cic/CIC_IDS2017.csv'
        elif data_template == 'kdd':
            full_data = './kdd/KDD.csv'
        else:
            raise Exception('The dataset template is not regcognize (cic or kdd)'
                            )
        main_df = pd.read_csv(full_data, low_memory=False, skiprows=progress_bar())

        main_df = scaler(main_df)
        
        X_main = main_df.drop('label', axis=1)
        y_main = main_df['label']

        # Split the main dataset
        X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(X_main, y_main, test_size=0.3, random_state=42, stratify=y_main)

        # Get the indices of the training and testing sets
        train_index = X_train_main.index
        test_index = X_test_main.index

        train_test_folder = [f'{data_template}/train_test_folder/train_{data_template}',
                            f'{data_template}/train_test_folder/test_{data_template}']
        
        train_combined = pd.concat([X_train_main, y_train_main], axis=1)
        test_combined = pd.concat([X_test_main, y_test_main], axis=1)
        
        train_combined.to_csv(f'.//{train_test_folder[0]}//train.csv', index=True)
        test_combined.to_csv(f'.//{train_test_folder[1]}//test.csv', index=True)
        

        print(f'Using Multiprocessing with : {num_processes}')
        try:
            for dataset_path in tqdm(dataset_paths, desc="Dataset paths"):
                
                print(f'== reading {dataset_path} ==')
                df = pd.read_csv(dataset_path, skiprows=progress_bar())
                # Splitting data
                X = df.drop('label', axis=1)
                y = df['label']

                sub_X_train = X.loc[train_index]
                sub_y_train = y.loc[train_index]
                sub_X_test = X.loc[test_index]
                sub_y_test = y.loc[test_index]
                
                # # Concatenate X_train with y_train, and X_test with y_test
                # train_combined = pd.concat([sub_X_train, sub_y_train], axis=1)
                # test_combined = pd.concat([sub_X_test, sub_y_test], axis=1)
                
                del df
                del X
                del y
                gc.collect()
                # Train and evaluate models on the current dataset
                results = {}
                
                dataset_name = dataset_path.split('\\')[-1]
                dataset_name = dataset_name.split('.')[0]
                print(f'dataset_name : {dataset_name}')
                if multiCPU:
                    # multiprocessing pool
                    args_list = [
                                    (name, model, data_template, dataset_name, sub_X_train, sub_y_train, sub_X_test, sub_y_test)
                                    for name, model in models.items()
                                ]
                    combined_results = {}
                    
                    with Pool(processes=num_processes) as pool:
                        results = pool.map(train_and_evaluate_Multiprocess, tqdm(args_list, desc=f"Training {data_template} Models"))

                        for result, arg in zip(results, args_list):
                            if result is not None:
                                # Unpack your args to get the model name and dataset name
                                name, _, _, dataset_name, _, _, _, _ = arg
                                combined_results[f"{dataset_name}_{name}"] = result
                            else:
                                # Log or print that this task was skipped
                                _, _, _, dataset_name, _, _, _, _ = arg
                                print(f"Skipped model for dataset: {dataset_name} due to ill-defined covariance.")

                    for result in results:
                        if result is not None:
                            for model_name, metrics in result.items():
                                combined_results[model_name] = {
                                    'accuracy': metrics[0],
                                    'loss': metrics[1],
                                    'f1': metrics[2],
                                    'precision': metrics[3],
                                    'recall': metrics[4],
                                    'confusion_matrix': metrics[5],  # Note that this will store the array in the DataFrame, which may not be what you want
                                }
                    
                    result_df = pd.DataFrame.from_dict(combined_results, orient='index', columns=['accuracy', 'loss', 'f1', 'precision', 'recall', 'confusion_matrix'])
                    result_filename = f"{data_template}/Training/compare/evaluation_results_{dataset_name}.csv"
                    result_df.to_csv(result_filename)

                    gc.collect()
                    
                else:
                    print('Using single CPU')
                    data = [sub_X_train, sub_X_test, sub_y_train, sub_y_test]
                    train_and_evaluation_singleprocess(models, data_template, data, dataset_name, results)
                    gc.collect()
                
        except ValueError as ve:
            if "covariance is ill defined" in str(ve):
                traceback.print_exc()
                print("Skipping due to ill-defined covariance.")
                
        print('== All training and evaluation is done ==')
        #Assemble the results
        compare_data = glob.glob(f'./{data_template}/Training/compare/*.csv')
        compare_df = best_model_for_attack(compare_data)
        compare_df.to_csv(f'{data_template}/model.csv')
        
    except Exception as E:
        print("An unexpected error occurred:", E)
        traceback.print_exc()



if __name__ == '__main__':
    main()