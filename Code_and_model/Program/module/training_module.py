
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



def process_data(Data, train_index, test_index, window_size):
    def rearrange_sequences(generator, index, window_size):
        rearranged_data = []

        for idx in tqdm(index, desc="Processing sequences"):
            if idx >= window_size - 1:  # Ensure index has enough preceding samples for a full sequence
                sequence_end = idx + 1  # Sequence ends at the current index (inclusive)
                sequence_start = sequence_end - window_size  # Sequence starts 'window_size' samples before the end

                batch_x = generator.data[sequence_start:sequence_end]  # Extract feature sequence
                batch_y = generator.targets[idx]  # Corresponding label

                rearranged_data.append((batch_x, batch_y))

        return rearranged_data
    # Process training data
    print('Training data processing...')
    train_data = rearrange_sequences(Data, train_index, window_size)
    gc.collect()
    
    print('Testing data processing...')
    test_data = rearrange_sequences(Data, test_index, window_size)
    del rearrange_sequences
    gc.collect()
    return train_data, test_data
    
def separate_features_labels(data, window_size):
    features = np.array([item[0] for item in data], dtype=np.float32)  # item[0] is each sequence
    labels = np.array([item[1] for item in data], dtype=np.float32)  # item[1] is the corresponding label

    # Ensure features are reshaped to (number_of_sequences, window_size, number_of_features_per_timestep)
    features = features.reshape(-1, window_size, features.shape[-1])

    return features, labels

    
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
        model.save(f'{data_template}/Training/model/{dataset_name}/{model.name}.h5')
        
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
        model.load_weights(f'{data_template}/Training/model/{dataset_name}/{model.name}.h5')

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
        result_filename = f"./{data_template}/Training/compare/evaluation_results_{dataset_name}"
        result_df.to_csv(result_filename)
        gc.collect()