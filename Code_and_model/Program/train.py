import os
import traceback
import argparse
import glob
import gc
import warnings

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from sklearn.model_selection import train_test_split

import tensorflow as tf



from module.model import getModel, sequential_models
from module.util import progress_bar, check_data_template
from module.file_op import *
from module.discord import *
from module.training_module import *


warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

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
        
        window_size = 512
        batch_size = 128
        epochs = 20
        
        DL_args = [window_size, batch_size, epochs]
        
        dataset_paths = glob.glob(f'{data_template}/dataset/mix_dataset/*.csv')
        
        check_data_template(data_template)
        
        if data_template == 'cic':
            full_data = './cic/CIC_IDS2017.csv'
        elif data_template == 'kdd':
            full_data = './kdd/KDD.csv'
        else:
            raise Exception('The dataset template is not regcognize (cic or kdd)')
            
        ## Process data for ML training
        main_df = pd.read_csv(full_data, low_memory=False, skiprows=progress_bar())
        
        X_main = main_df.drop('label', axis=1)
        y_main = main_df['label']
        n_features = X_main.shape[1]
        
        X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(X_main, y_main, test_size=0.3, random_state=42, stratify=y_main)

        
        train_index = X_train_main.index
        test_index = X_test_main.index

        train_test_folder = [f'{data_template}/train_test_folder/train_{data_template}',
                            f'{data_template}/train_test_folder/test_{data_template}']
        
        train_combined = pd.concat([X_train_main, y_train_main], axis=1)
        test_combined = pd.concat([X_test_main, y_test_main], axis=1)
        
        train_combined.to_csv(f'.//{train_test_folder[0]}//train.csv', index=True)
        test_combined.to_csv(f'.//{train_test_folder[1]}//test.csv', index=True)
        
        models = getModel()
        sequence_models = sequential_models(window_size, n_features)
        
        
        ## ML model Training
        print(f'Using Multiprocessing with : {num_processes}')
        try:
            for dataset_path in tqdm(dataset_paths, desc="Dataset paths"):
                
                print(f'== reading {dataset_path} ==')
                df = pd.read_csv(dataset_path, skiprows=progress_bar())
                X = df.drop('label', axis=1)
                y = df['label']

                sub_X_train = X.loc[train_index]
                sub_y_train = y.loc[train_index]
                sub_X_test = X.loc[test_index]
                sub_y_test = y.loc[test_index]

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
                                name, _, _, dataset_name, _, _, _, _ = arg
                                combined_results[f"{dataset_name}_{name}"] = result
                            else:
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
                                    'confusion_matrix': metrics[5],
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

        
        ## DL model Training
        
        try:
            print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
            
            train_index = pd.read_csv(f'./{data_template}/train_test_folder/train_{data_template}/train.csv')['Unnamed: 0']
            test_index = pd.read_csv(f'./{data_template}/train_test_folder/train_{data_template}/test.csv')['Unnamed: 0']
            train_test_index = [train_index, test_index]
            
            for dataset_path in tqdm(dataset_paths, desc="Dataset paths"):
                print(f'== reading {dataset_path} ==')
                df = pd.read_csv(dataset_path, skiprows=progress_bar())
                
                results = {}

                dataset_name = dataset_path.split('\\')[-1]
                dataset_name = dataset_name.split('.')[0]
                print(f'dataset_name : {dataset_name}')

                training_DL(sequence_models, data_template, dataset_name, results, df, DL_args, train_test_index)
                
        except ValueError as ve:
            print(ve)

        compare_data = glob.glob(f'./{data_template}/Training/compare/*.csv')
        compare_df = best_model_for_attack(compare_data)
        compare_df.to_csv(f'{data_template}/model.csv')
        
    except Exception as E:
        print("An unexpected error occurred:", E)
        traceback.print_exc()



if __name__ == '__main__':
    main()