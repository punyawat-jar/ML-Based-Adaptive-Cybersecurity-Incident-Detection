
import os
import gc

import joblib
from joblib import dump

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
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