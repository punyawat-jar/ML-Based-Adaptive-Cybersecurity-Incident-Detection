import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import glob
import requests
import json
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from joblib import dump


def makePath(path):
    if not os.path.exists(path):
        os.makedirs(path)
def progress_bar(*args, **kwargs):
    bar = tqdm(*args, **kwargs)

    def checker(x):
        bar.update(1)
        return False

    return checker



def makePath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    os.chdir("C:\\Users\\Kotani Lab\\Desktop\\ML_senior_project\\ML-Based-Adaptive-Cybersecurity-Incident-Detection\\Code_and_model\\cic")

    models = {
        'LogisticRegression': LogisticRegression(max_iter=10000, n_jobs=-1),
        'ExtraTrees': ExtraTreesClassifier(n_jobs=-1),
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

    dataset_paths = glob.glob('.\\dataset\\mix_dataset\\*.csv')

    backslash = "\\"
    
    train_indices = pd.read_csv('../Program\\cic\\train_test_folder\\train_cic\\train.csv')['Unnamed: 0']
    test_indices = pd.read_csv('../Program\\cic\\train_test_folder\\test_cic\\test.csv')['Unnamed: 0']

    # Split the main dataset
    saveTrain_test = './/train_test_folder'
    train_dir = './train_test_folder/train_cic'
    test_dir ='./train_test_folder/test_cic'
    makePath(saveTrain_test)
    makePath(train_dir)
    makePath(test_dir)

    for dataset_path in tqdm(dataset_paths, desc="Dataset paths"):

        print(f'== reading {dataset_path} ==')
        df = pd.read_csv(dataset_path, low_memory=False, skiprows=progress_bar())

        X = df.drop('label', axis=1)
        y = df['label']

        sub_X_train, sub_X_test = X.loc[train_indices], X.loc[test_indices]
        sub_y_train, sub_y_test = y.loc[train_indices], y.loc[test_indices]

        del df
        del X
        del y
        gc.collect()
        results = {}
        dataset_name = dataset_path.split(backslash)[-1]  # Name of the dataset

        for name, model in tqdm(models.items(), desc="Training CIC Models"):

            try:
                print(f'== Mix CIC Training: {dataset_name} with model: {name} ==')
                
                model.fit(sub_X_train, sub_y_train)
                y_pred = model.predict(sub_X_test)

                accuracy = accuracy_score(sub_y_test, y_pred)
                f1 = f1_score(sub_y_test, y_pred)
                precision = precision_score(sub_y_test, y_pred)
                recall = recall_score(sub_y_test, y_pred)
                conf_matrix = confusion_matrix(sub_y_test, y_pred, labels=model.classes_)
                
                conf_matrix_path = f'.\\classical_ML\\mix_training\\confusion_martix\\{dataset_name}'

                if not os.path.exists(conf_matrix_path):
                    os.makedirs(conf_matrix_path)
                    
                cm_dis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
                fig, ax = plt.subplots()
                cm_dis.plot(ax=ax)
                fig.savefig(f".\\classical_ML\\mix_training\\confusion_martix\\{dataset_name}\\{dataset_name}_{name}_confusion_matrix.png")

                # Close the figure
                plt.close(fig)


                loss = np.mean(np.abs(y_pred - sub_y_test))
                print(f'== Done Training: {dataset_name} with model: {name}, acc: {accuracy}, loss: {loss}, f1: {f1} ==')
                models_save_path = f".\\classical_ML\\mix_training\\model\\{dataset_name}"
                if not os.path.exists(models_save_path):
                    os.makedirs(models_save_path)

                # Save the trained model
                model_filename =  f".\\classical_ML\\mix_training\\model\\{dataset_name}\\{dataset_name}_{name}_model.joblib"
                dump(model, model_filename)
                print(f"== CIC Model {name} saved as {model_filename} ==")
                
                results[name] = [accuracy, loss, f1, precision, recall, conf_matrix]

            except Exception as E:
                print(f'Error : {E}')

        # Convert results to DataFrame and save with dataset name
        shutil.move(f'{dataset_path}', f'.\\dataset\\\\mix_done\\{dataset_name}')
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['accuracy', 'loss', 'f1', 'precision', 'recall', 'confusion_matrix'])
        result_filename = f".\\classical_ML\\mix_training\\compare\\evaluation_results_{dataset_name}"
        result_df.to_csv(result_filename)
        gc.collect()
    print('== @everyone All training and evaluation is done ==')


if __name__ == '__main__':
    main()