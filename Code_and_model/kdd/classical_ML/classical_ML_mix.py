import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import glob
import requests
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from joblib import dump
os.chdir("C:\\Users\\Kotani Lab\\Desktop\\ML_senior_project\\ML-Based-Adaptive-Cybersecurity-Incident-Detection\\Code_and_model\\kdd")
botnum = 1
bot = ['https://discord.com/api/webhooks/1162767976034996274/B6CjtQF1SzNRalG_csFx8-qJ5ODBoy5SBUelbGyl-v-QhYhwdsTfE59F-K-rXj3HyUh-',
      'https://discord.com/api/webhooks/1162767979658887299/0TICfekiC9wjPmp-GqE5zrwU57q2RJHG2peel_KOYagUDYCjovYUfyNJmDR9jbD-WXoE']

def processlabel(df):
    df.loc[df['label'] == 'normal', 'label'] = 0
    df.loc[df['label'] != 0, 'label'] = 1
    df['label'] = df['label'].astype('int')
    return df
def makePath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def preprocess(df):
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df

def send_discord_message(content):
    webhook_url = bot[botnum]

    data = {
        'content': content
    }

    response = requests.post(webhook_url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

    if response.status_code != 204:
        raise ValueError(f'Request to discord returned an error {response.status_code}, the response is:\n{response.text}')
        

def train_mixed():
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
    dataset_paths = glob.glob('.\\dataset\\mix_dataset\\*.csv')

    # Lists to store metrics
    f1_scores = []
    precision_scores = []
    recall_scores = []
    cms = []
    FPRs = []
    FNRs = []
    TPRs = []
    TNRs = []
    backslash = "\\"



    saveTrain_test = './/train_test_folder'
    train_dir = './train_test_folder/train_kdd'
    test_dir ='./train_test_folder/test_kdd'
    makePath(saveTrain_test)
    makePath(train_dir)
    makePath(test_dir)


    for train_path in tqdm(dataset_paths, desc="Dataset paths"):
        print(f"== reading : {train_path} ==")
        df = pd.read_csv(train_path)

        X = df.drop('label', axis=1)
        y = df['label']
        
        sub_X_train, sub_X_test, sub_y_train, sub_y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify= y)
            
        print(y.value_counts())
        results = {}
        
        dataset_name = train_path.split(backslash)[-1]  # Name of the dataset
        
        for name, model in tqdm(models.items(), desc="Training KDD Models"):
            try:
                # send_discord_message(f'== Mix CIC Training: {dataset_name} with model: {name} ==')
                print(f'== Mix KDD Training: {dataset_name} with model: {name} ==')
                
                model.fit(sub_X_train, sub_y_train)
                y_pred = model.predict(sub_X_test)
        
                accuracy = accuracy_score(sub_y_test, y_pred)
                f1 = f1_score(sub_y_test, y_pred, zero_division = 1)
                precision = precision_score(sub_y_test, y_pred, zero_division = 1)
                recall = recall_score(sub_y_test, y_pred, zero_division = 1)
                conf_matrix = confusion_matrix(sub_y_test, y_pred, labels=model.classes_)
                
                print(conf_matrix)
                conf_matrix_path = f".\\classical_ML\\mix_training\\confusion_martix\\{train_path.split(backslash)[-1]}"
                if not os.path.exists(conf_matrix_path):
                    os.makedirs(conf_matrix_path)
                
                # Your existing code for confusion matrix
                cm_dis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
                fig, ax = plt.subplots()
                cm_dis.plot(ax=ax)
                fig.savefig(f".\\classical_ML\\mix_training\\confusion_martix\\{dataset_name}\\{dataset_name}_{name}_confusion_matrix.png")
                plt.close(fig)

                loss = np.mean(np.abs(y_pred - sub_y_test))
                print(f"== Done Training: {train_path.split(backslash)[-1]} with model: {name}, acc: {accuracy}, loss: {loss}, f1: {f1} ==")
                models_save_path = f".\\classical_ML\\mix_training\\model\\{train_path.split(backslash)[-1]}"
                if not os.path.exists(models_save_path):
                    os.makedirs(models_save_path)
        
                model_filename = f".\\classical_ML\\mix_training\\model\\{train_path.split(backslash)[-1]}\\{train_path.split(backslash)[-1]}_{name}_model.joblib"
                dump(model, model_filename)
                print(f"== Model {name} saved as {model_filename} ==")
                
                results[name] = [accuracy, loss, f1, precision, recall, conf_matrix]
            except Exception as error:
                print(f'Error : {error}')

        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['accuracy', 'loss', 'f1', 'precision', 'recall', 'confusion_matrix'])
        result_filename = f".\\classical_ML\\mix_training\\compare\\evaluation_results_{train_path.split(backslash)[-1]}"
        result_df.to_csv(result_filename)
        
    print('== @everyone All training and evaluation is done ==')
    
    
if __name__ == '__main__':
    train_mixed()