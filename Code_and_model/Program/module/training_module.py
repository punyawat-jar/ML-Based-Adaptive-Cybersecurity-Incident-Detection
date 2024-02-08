
import os
import gc

import joblib
from joblib import dump

import numpy as np

from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt


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
        
        models_save_path = f'{data_template}/Training/model/{dataset_name}'
        
        if not os.path.exists(models_save_path):
            os.makedirs(models_save_path)

        model_filename = f"{models_save_path}/{dataset_name}_{name}_model.joblib"
        dump(model, model_filename)
    
    except ValueError as ve:
        if "covariance is ill defined" in str(ve):
            print("Skipping due to ill-defined covariance for dataset:", dataset_name)
            return None
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
        

        model_filename =  f"{data_template}/Training/model/{dataset_name}/{dataset_name}_{name}_model.joblib"
        dump(model, model_filename) 

        results[name] = [accuracy, loss, f1, precision, recall, conf_matrix]
        
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['accuracy', 'loss', 'f1', 'precision', 'recall', 'confusion_matrix'])
        result_filename = f"{data_template}/Training/compare/evaluation_results_{dataset_name}"
        result_df.to_csv(result_filename)
        gc.collect()
        
def rearrange_sequences_linear_chunked(generator, df_index, batch_size, window_size, chunk_size=100):
    num_batches = len(generator)
    
    rearranged_data = []  # This will store the final results

    for chunk_start in tqdm(range(0, num_batches, chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, num_batches)
        for i in range(chunk_start, chunk_end):
            batch_x, batch_y = generator[i]  # Access each batch from the generator

            for j in range(batch_x.shape[0]):  # Process each sequence in the batch
                sequence_start_index = i * batch_size + j
                original_indices = df_index[sequence_start_index: sequence_start_index + window_size].tolist()
                rearranged_data.append((batch_x[j], batch_y[j], original_indices))
        
        gc.collect()  # Suggest to the garbage collector to release unreferenced memory

    return rearranged_data

def training_DL(models, data_template, data, dataset_name, results, epochs, df):
    X = df.drop('label', axis=1)
    y = df['label']
    
    Data = TimeseriesGenerator(X, y, length=window_size, sampling_rate=1, batch_size=batch_size)
    
    rearranged_data = rearrange_sequences_linear_chunked(Data, df['Unnamed: 0'], batch_size, window_size, chunk_size=50)
    
    for name, model in models.items():
        models_save_path = f'{data_template}/Training/model/{dataset_name}'
        conf_matrix_path = f'{data_template}/Training/confusion_martix/{dataset_name}'
        checkpoint_path = f'{data_template}/Training/checkpoint/{dataset_name}'

        makePath(models_save_path)
        makePath(conf_matrix_path)
        makePath(checkpoint_path)

        earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='min')
        best_model = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

        model.fit(X, y, epochs, verbose=0)
        model.save(f'./{data_template}/model/{model.name}.h5')

        history = model.fit(
            data,
            epochs=epochs,
            callbacks=[best_model, earlyStopping],
            validation_data=val_dataset,
            validation_steps=validation_steps
        )
        
        results[name] = [accuracy, loss, f1, precision, recall, conf_matrix]
        
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['accuracy', 'loss', 'f1', 'precision', 'recall', 'confusion_matrix'])
        result_filename = f"{data_template}/Training/compare/evaluation_results_{dataset_name}"
        result_df.to_csv(result_filename)
        gc.collect()