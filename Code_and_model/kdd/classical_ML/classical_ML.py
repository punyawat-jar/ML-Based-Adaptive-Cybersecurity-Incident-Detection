import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import glob
import requests
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from joblib import dump


def train_classical():

    os.chdir("C:\\Users\\Kotani Lab\\Desktop\\ML_senior_project\\ML-Based-Adaptive-Cybersecurity-Incident-Detection\\Code_and_model\\kdd")

    models = {
        'LogisticRegression': LogisticRegression(max_iter=10000, n_jobs=-1),
        'ExtraTrees': ExtraTreesClassifier(n_jobs=-1),
        'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), n_jobs=-1),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(),
        'KNeighbors': KNeighborsClassifier(n_jobs=-1),
        'GaussianNB': GaussianNB(),
        'Perceptron': Perceptron(n_jobs=-1),
        'AdaBoost': AdaBoostClassifier()
    }

    dataset_paths = glob.glob('./dataset/normal_dataset/*.csv')

    backslash = "\\"
    datasets = {train_path.split(backslash)[-1]: (train_path, train_path) for train_path in dataset_paths}

    
    
    for dataset_name, (train_path, _) in datasets.items():
        print(f"== reading training data: {train_path} ==")
        df = pd.read_csv(train_path)
        
        X = df.drop('label', axis=1)
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify= y)
        
        print(y.value_counts())
        results = {}
        
        for name, model in models.items():
            try:
                print(f"== Training: {train_path.split(backslash)[-1]} with model: {name} ==")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
        
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, zero_division = 1)
                precision = precision_score(y_test, y_pred, zero_division = 1)
                recall = recall_score(y_test, y_pred, zero_division = 1)
                conf_matrix = confusion_matrix(y_test, y_pred)
                if conf_matrix.size == 1:
                    TN, FP, FN, TP = 0, 0, 0, conf_matrix[0][0]
                else:
                    TN, FP, FN, TP = conf_matrix.ravel()
                conf_matrix_path = f"./classical_ML/label_training/confusion_martix/{train_path.split(backslash)[-1]}"
                if not os.path.exists(conf_matrix_path):
                    os.makedirs(conf_matrix_path)
                    
                plt.figure()
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                plt.savefig(f"./classical_ML/label_training/confusion_martix/{train_path.split(backslash)[-1]}/{train_path.split(backslash)[-1]}_{name}_confusion_matrix.png")
                plt.close()
        
                loss = np.mean(np.abs(y_pred - y_test))
                print(f"== Done Training: {train_path.split(backslash)[-1]} with model: {name}, acc: {accuracy}, loss: {loss}, f1: {f1} ==")
                models_save_path = f"./classical_ML/label_training/model/{train_path.split(backslash)[-1]}"
                if not os.path.exists(models_save_path):
                    os.makedirs(models_save_path)
        
                model_filename = os.path.join(models_save_path, f"{train_path.split(backslash)[-1]}_{name}_model.joblib")
                dump(model, model_filename)
                print(f"== Model {name} saved as {model_filename} ==")
                
                results[name] = [accuracy, loss, f1, precision, recall, conf_matrix]
            except Exception as error:
                print(f'Error : {error}')

        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['accuracy', 'loss', 'f1', 'precision', 'recall', 'confusion_matrix'])
        result_filename = f"./classical_ML/label_training/compare/evaluation_results_{train_path.split(backslash)[-1]}"
        result_df.to_csv(result_filename)
        
    print('== @everyone All training and evaluation is done ==')
    
if __name__ == '__main__':
    train_classical()