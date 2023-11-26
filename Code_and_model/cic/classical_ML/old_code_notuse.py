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
os.chdir('/home/s2316002/capstone_project/cic/classical_ML')
botnum = 0
bot = ['https://discord.com/api/webhooks/1162767976034996274/B6CjtQF1SzNRalG_csFx8-qJ5ODBoy5SBUelbGyl-v-QhYhwdsTfE59F-K-rXj3HyUh-',
      'https://discord.com/api/webhooks/1162767979658887299/0TICfekiC9wjPmp-GqE5zrwU57q2RJHG2peel_KOYagUDYCjovYUfyNJmDR9jbD-WXoE']

def preprocess(df):
    df.loc[df['label'] == "BENIGN", "label"] = 0
    df.loc[df['label'] != 0, "label"] = 1

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

models = {
    'LogisticRegression': LogisticRegression(max_iter=10000, n_jobs=-1),
    # 'LinearSVM': SVC(kernel="linear", probability=True),  # SVC does not support n_jobs
    # 'RBFSVM': SVC(kernel="rbf", probability=True),  # SVC does not support n_jobs
    'ExtraTrees': ExtraTreesClassifier(n_jobs=-1),
    # 'Bagging': BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_jobs=-1),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'DecisionTree': DecisionTreeClassifier(),  # Single decision tree does not support n_jobs
    'RandomForest': RandomForestClassifier(n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(),  # GradientBoosting does not support n_jobs
    'KNeighbors': KNeighborsClassifier(n_jobs=-1),
    'GaussianNB': GaussianNB(),  # GaussianNB does not support n_jobs
    'Perceptron': Perceptron(n_jobs=-1),
    'AdaBoost': AdaBoostClassifier()  # AdaBoost does not support n_jobs
}

dataset_paths = glob.glob('/home/s2316002/capstone_project/cic/dataset/label_dataset/*.csv')

# Lists to store metrics
f1_scores = []
precision_scores = []
recall_scores = []
cms = []
FPRs = []
FNRs = []
TPRs = []
TNRs = []

for dataset_path in dataset_paths:
    # Load and preprocess dataset
    print(f'== reading {dataset_path} ==')
    df = pd.read_csv(dataset_path)
    df = preprocess(df)
    print(f'== Done reading {dataset_path} ==')
    # Splitting data
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train and evaluate models on the current dataset
    results = {}
    dataset_name = dataset_path.split('/')[-1]  # Name of the dataset
    
    for name, model in models.items():
        send_discord_message(f'== CIC Training: {dataset_name} with model: {name} ==')
        print(f'== CIC Training: {dataset_name} with model: {name} ==')
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Extract metrics from confusion matrix
        TN, FP, FN, TP = conf_matrix.ravel()
        
        conf_matrix_path = f'/home/s2316002/capstone_project/cic/classical_ML/confusion_martix/{dataset_name}'
        if not os.path.exists(conf_matrix_path):
            os.makedirs(conf_matrix_path)
            
        # Plot and save confusion matrix
        plt.figure()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(f'/home/s2316002/capstone_project/cic/classical_ML/confusion_martix/{dataset_name}/{dataset_name}_{name}_confusion_matrix.png')
        plt.close()

        # Here I assume binary classification for loss (0 or 1). Adjust if needed.
        loss = np.mean(np.abs(y_pred - y_test))
        send_discord_message(f'== CIC Done Training: {dataset_name} with model: {name}, acc: {accuracy}, loss: {loss}, f1: {f1} ==')
        print(f'== Done Training: {dataset_name} with model: {name}, acc: {accuracy}, loss: {loss}, f1: {f1} ==')
        models_save_path = f"/home/s2316002/capstone_project/cic/classical_ML/model/{dataset_name}"
        if not os.path.exists(models_save_path):
            os.makedirs(models_save_path)

        # Save the trained model
        model_filename = os.path.join(models_save_path, f"/home/s2316002/capstone_project/cic/classical_ML/model/{dataset_name}/{dataset_name}_{name}_model.joblib")
        dump(model, model_filename)
        print(f"== CIC Model {name} saved as {model_filename} ==")
        
        results[name] = [accuracy, loss, f1, precision, recall, conf_matrix]

    # Convert results to DataFrame and save with dataset name
    result_df = pd.DataFrame.from_dict(results, orient='index', columns=['accuracy', 'loss', 'f1', 'precision', 'recall', 'confusion_matrix'])
    result_filename = f"/home/s2316002/capstone_project/cic/classical_ML/compare/evaluation_results_{dataset_name}"
    result_df.to_csv(result_filename)
    
# send_discord_message('== @everyone All training and evaluation is done ==')
print('== @everyone All training and evaluation is done ==')
