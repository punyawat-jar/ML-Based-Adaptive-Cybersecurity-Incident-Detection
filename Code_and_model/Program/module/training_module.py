
import os
import gc

import joblib
from joblib import dump
import datetime
import numpy as np

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

from module.file_op import *
from module.discord import *
from module.model import getModel
from module.util import *
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)



class joblib_model:
    def __init__(self, model, weight):
        self.model = joblib.load(model)
        self.weight = weight

def best_model_for_attack(model_folder):            ## If the model's evaluation f1 and acc are equal, then use the priority model from user input 
    bestmodel = {
        'attack': [],
        'model': [],
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': []
    }
    
    model_priority = list(getModel().keys())
    
    for model in model_folder:
        modelname = model.split('_')[-1].split('.')[0]
        df = pd.read_csv(model).sort_values(['f1', 'accuracy'], ascending=[False, False])

        if df.empty:
            append_default_values(bestmodel, modelname)
        else:
            if len(df) > 1 and df.iloc[0]['f1'] == df.iloc[1]['f1'] and df.iloc[0]['accuracy'] == df.iloc[1]['accuracy']:
                
                top_models = df[(df['f1'] == df.iloc[0]['f1']) & (df['accuracy'] == df.iloc[0]['accuracy'])]
                selected_model = select_model_based_on_priority(top_models, model_priority)
            else:
                selected_model = df.iloc[0]
            
            append_model_data(bestmodel, modelname, selected_model)

    return pd.DataFrame(data=bestmodel)

def append_default_values(bestmodel, modelname):
    bestmodel['attack'].append(modelname)
    bestmodel['model'].append(None)
    bestmodel['accuracy'].append(None)
    bestmodel['f1'].append(None)
    bestmodel['precision'].append(None)
    bestmodel['recall'].append(None)
    
def append_model_data(bestmodel, modelname, selected_model):
    bestmodel['attack'].append(modelname)
    bestmodel['model'].append(selected_model['model'])
    bestmodel['accuracy'].append(selected_model['accuracy'])
    bestmodel['f1'].append(selected_model['f1'])
    bestmodel['precision'].append(selected_model['precision'])
    bestmodel['recall'].append(selected_model['recall'])

def select_model_based_on_priority(top_models, model_priority):
    for priority_model in model_priority:
        if priority_model in top_models['model'].values:
            return top_models[top_models['model'] == priority_model].iloc[0]
    return top_models.iloc[0]


def save_model(model, path, args):
    name, model, dataset_name = args
    model_filename = f"{path}/{dataset_name}_{name}_model.joblib"
    dump(model, model_filename)


def train_and_evaluate_Multiprocess(args):
    try:
        name, model, data_template, dataset_name, sub_X_train, sub_y_train, sub_X_test, sub_y_test = args
        
        models_save_path = f'{data_template}/Training/model/{dataset_name}'
        conf_matrix_path = f'{data_template}/Training/confusion_matrix/{dataset_name}'
        makePath(models_save_path)
        makePath(conf_matrix_path)
        
        saving_args = [name, model, dataset_name]
        
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
        
        save_model(model, models_save_path, saving_args)

    except ValueError as ve:
        if "covariance is ill defined" in str(ve):
            print("Skipping due to ill-defined covariance for dataset:", dataset_name)
            return None
    
    return {name: [accuracy, loss, f1, precision, recall, conf_matrix]}

def train_and_evaluation_singleprocess(models, data_template, data, dataset_name, results):
    for name, model in models.items():
        sub_X_train, sub_X_test, sub_y_train, sub_y_test = data
        
        models_save_path = f'{data_template}/Training/model/{dataset_name}'
        conf_matrix_path = f'{data_template}/Training/confusion_matrix/{dataset_name}'
        makePath(models_save_path)
        makePath(conf_matrix_path)
        
        saving_args = [name, model, dataset_name]
        
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
        fig.savefig(f'{data_template}/Training/confusion_matrix/{dataset_name}/{dataset_name}_{name}_confusion_matrix.png')

        plt.close(fig)
        
        loss = np.mean(np.abs(y_pred - sub_y_test))
    
        results[name] = [accuracy, loss, f1, precision, recall, conf_matrix]
        
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['accuracy', 'loss', 'f1', 'precision', 'recall', 'confusion_matrix'])
        result_filename = f"{data_template}/Training/compare/evaluation_results_{dataset_name}"
        result_df.to_csv(result_filename)
        
        save_model(model, models_save_path, saving_args)
        gc.collect()

def update_evaluation_results(result_df, dataset_name, data_template):
    
    result_df.rename(columns={result_df.columns[0]: "model"}, inplace=True)
    # Construct the file path
    result_filename = f"{data_template}/Training/compare/evaluation_results_{dataset_name}.csv"
    
    # Check if the file exists
    if os.path.exists(result_filename):
        # Read the existing data
        existing_df = pd.read_csv(result_filename)
        
        # Update existing DataFrame with new values from result_df based on the "model" column
        for model in result_df['model']:
            for metric in ['accuracy', 'loss', 'f1', 'precision', 'recall', 'confusion_matrix']:
                existing_df.loc[existing_df['model'] == model, metric] = result_df.loc[result_df['model'] == model, metric].values[0]
        
        # Save the updated DataFrame back to CSV
        existing_df.to_csv(result_filename, index=False)
    else:
        # If the file doesn't exist, save result_df as a new CSV file
        result_df.to_csv(result_filename, index=False)

    print("Evaluation results updated successfully!")
    


    
def training_DL(models, data_template, dataset_name, df, DL_args, train_test_df, train_test_index):
    results = {}
    X = df.drop('label', axis=1)
    y = df['label']
    
    window_size, batch_size, epochs = DL_args
    
    Data = TimeseriesGenerator(X, y, length=window_size, sampling_rate=1)
    print(f'Data type {type(Data)}')
    train_index, test_index = train_test_index
    print(train_index)
    print(test_index)
    print('Processing Training data...')
    
    train_data, test_data = process_data(Data, train_index, test_index, window_size)
    gc.collect()
    X_train, y_train = separate_features_labels(train_data, window_size)
    X_test, y_test = separate_features_labels(test_data, window_size)
    
    for name, model in models.items():
        models_save_path = f'{data_template}/Training/model/{dataset_name}'
        conf_matrix_path = f'{data_template}/Training/confusion_matrix/{dataset_name}'
        checkpoint_path = f'{data_template}/Training/checkpoint/{dataset_name}'

        makePath(models_save_path)
        makePath(conf_matrix_path)
        makePath(checkpoint_path)

        earlyStopping = EarlyStopping(monitor='loss', patience=2, verbose=0, mode='min')
        best_model = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='loss', mode='min')

        log_dir = os.path.join(data_template, "Training", "logs", dataset_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir)
        print(f'Saving log path at: {log_dir}')
        model.save(f'{data_template}/Training/model/{dataset_name}/{dataset_name}_{model.name}_model.h5')
        
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        print(model.summary())
        steps_per_epoch = np.ceil(len(X_train) / batch_size)
        validation_steps = np.ceil(len(X_test) / batch_size)
        
        print(f'Steps per epoch: {steps_per_epoch}, vali: {validation_steps}')
        history = model.fit(
            x=X_train, 
            y=y_train,
            epochs=epochs,
            steps_per_epoch=len(X_train) // batch_size,
            verbose=1,
            callbacks=[best_model, earlyStopping, tensorboard_callback],
            validation_data=(X_test, y_test),
            validation_steps=len(X_test) // batch_size
        )

        # Load the best model for evaluation
        model.load_weights(f'{data_template}/Training/model/{dataset_name}/{dataset_name}_{model.name}_model.h5')

        # Evaluation on test data
        y_true, y_pred = [], []
        for i in range(0, len(test_index), batch_size):
            batch_indices = test_index[i:i + batch_size]
            batch_x, batch_y = [], []

            for idx in batch_indices:
                sequence_start = idx - window_size + 1
                sequence_end = idx + 1

                if sequence_start >= 0 and sequence_end <= len(Data.data):
                    x_sequence = Data.data[sequence_start:sequence_end]
                    y_sequence = Data.targets[idx]

                    batch_x.append(x_sequence)
                    batch_y.append(y_sequence)

            batch_x = np.array(batch_x)
            y_batch_pred = model.predict(batch_x)
            y_batch_pred_labels = np.argmax(y_batch_pred, axis=1)

            y_true.extend(batch_y)
            y_pred.extend(y_batch_pred_labels)


        # Compute metrics
        f1 = f1_score(y_true, y_pred, average='binary')  # Adjust 'average' as needed
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')

        conf_matrix = confusion_matrix(y_true, y_pred)

        
        cm_dis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        fig, ax = plt.subplots()
        cm_dis.plot(ax=ax)
        fig.savefig(f'{data_template}/Training/confusion_matrix/{dataset_name}/{dataset_name}_{name}_confusion_matrix.png')

        plt.close(fig)

        val_acc = history.history['val_accuracy']
        val_loss=history.history['val_loss']

        print(f'Finished Training {dataset_name} with : F1: {f1}, Acc: {val_acc}, Loss: {val_loss}')
        results[name] = [val_acc, val_loss, f1, precision, recall, conf_matrix]
        
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['accuracy', 'loss', 'f1', 'precision', 'recall', 'confusion_matrix'])
        update_evaluation_results(result_df, data_template, dataset_name)
        
        gc.collect()