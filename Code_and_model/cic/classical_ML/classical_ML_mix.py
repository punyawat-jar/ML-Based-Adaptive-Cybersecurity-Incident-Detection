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
os.chdir("C:\\Users\\Kotani Lab\\Desktop\\ML_senior_project\\ML-Based-Adaptive-Cybersecurity-Incident-Detection\\Code_and_model\\cic")
botnum = 0
bot = ['https://discord.com/api/webhooks/1132920901235654676/691VuY4nCL4yTjkqJHAtG6u3oXUxRonIxulHx3i3chJO92G0Ug6XxtdIyWVqyDk4vDLW',
      'https://discord.com/api/webhooks/1133199528284135515/uRBbJul9XFEA9YPqnvDSpZsQvSauZMzMdoBFnb8q69ILE_wVrxqxhkdDTeb-smGBmgIo']

def makePath(path):
    if not os.path.exists(path):
        os.makedirs(path)
def progress_bar(*args, **kwargs):
    bar = tqdm(*args, **kwargs)

    def checker(x):
        bar.update(1)
        return False

    return checker

def preprocess(df):
    
    df.loc[df['label'] == "BENIGN", "label"] = 0
    df.loc[df['label'] != 0, "label"] = 1

    scaler = MinMaxScaler()
    for col in df.columns:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).ravel()

    return df

def makePath(path):
    if not os.path.exists(path):
        os.makedirs(path)


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

dataset_paths = glob.glob('.\\mix_dataset\\*.csv')

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

main_df = pd.read_csv('./CIC_IDS2017.csv', low_memory=False, skiprows=progress_bar())

X_main = main_df.drop('label', axis=1)
y_main = main_df['label']

# Split the main dataset
X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(X_main, y_main, test_size=0.3, random_state=42, stratify=y_main)

main_train_combined = pd.concat([X_train_main, y_train_main], axis=1)
main_test_combined = pd.concat([X_test_main, y_test_main], axis=1)



    
# Get the indices of the training and testing sets
train_index = X_train_main.index
test_index = X_test_main.index

saveTrain_test = './/train_test_folder'
train_dir = './train_test_folder/train_cic'
test_dir ='./train_test_folder/test_cic'
makePath(saveTrain_test)
makePath(train_dir)
makePath(test_dir)

main_train_combined.to_csv(f'.//{train_dir}//train_main.csv', index=False)
main_test_combined.to_csv(f'.//{test_dir}//test_main.csv', index=False)

for dataset_path in tqdm(dataset_paths, desc="Dataset paths"):
    # Load and preprocess dataset
    print(f'== reading {dataset_path} ==')
    df = pd.read_csv(dataset_path, low_memory=False, skiprows=progress_bar())
    # Splitting data
    X = df.drop('label', axis=1)
    y = df['label']

    sub_X_train = X.loc[train_index]
    sub_y_train = y.loc[train_index]
    sub_X_test = X.loc[test_index]
    sub_y_test = y.loc[test_index]
    
    # Concatenate X_train with y_train, and X_test with y_test
    train_combined = pd.concat([sub_X_train, sub_y_train], axis=1)
    test_combined = pd.concat([sub_X_test, sub_y_test], axis=1)

    del df
    del X
    del y
    gc.collect()
    # Train and evaluate models on the current dataset
    results = {}
    dataset_name = dataset_path.split(backslash)[-1]  # Name of the dataset

    #To be debugged, deleted if the same
    #======
    train_combined.to_csv(f'.//{train_dir}//train_{dataset_name}.csv', index=False)
    test_combined.to_csv(f'.//{test_dir}//test_{dataset_name}.csv', index=False)
    #======

    for name, model in tqdm(models.items(), desc="Training CIC Models"):

        try:
            send_discord_message(f'== Mix CIC Training: {dataset_name} with model: {name} ==')
            print(f'== Mix CIC Training: {dataset_name} with model: {name} ==')
            
            model.fit(sub_X_train, sub_y_train)
            y_pred = model.predict(sub_X_test)

            accuracy = accuracy_score(sub_y_test, y_pred)
            f1 = f1_score(sub_y_test, y_pred)
            precision = precision_score(sub_y_test, y_pred)
            recall = recall_score(sub_y_test, y_pred)
            conf_matrix = confusion_matrix(sub_y_test, y_pred, labels=model.classes_)

            # Extract metrics from confusion matrix
            TN, FP, FN, TP = conf_matrix.ravel()
            
            conf_matrix_path = f'.\\classical_ML\\mix_training\\confusion_martix\\{dataset_name}'

            if not os.path.exists(conf_matrix_path):
                os.makedirs(conf_matrix_path)
                
            # Plot and save confusion matrix
            # Your existing code for confusion matrix
            cm_dis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
            fig, ax = plt.subplots()
            cm_dis.plot(ax=ax)
            fig.savefig(f".\\classical_ML\\mix_training\\confusion_martix\\{dataset_name}\\{dataset_name}_{name}_confusion_matrix.png")

            # Close the figure
            plt.close(fig)
            
            # plt.figure(figsize=(10, 8))
            # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
            # plt.xlabel('Predicted', fontsize=14)
            # plt.ylabel('Actual', fontsize=14)
            # plt.title(f'Confusion Matrix of {dataset_name}', fontsize=18)
            # plt.tight_layout()  # Adjust the padding of the plot to fit everything
            # plt.savefig(f'.\\classical_ML\\mix_training\\confusion_martix\\{dataset_name}\\{dataset_name}_{name}_confusion_matrix.png')
            # plt.close()
            # plt.show()

            # Here I assume binary classification for loss (0 or 1). Adjust if needed.
            loss = np.mean(np.abs(y_pred - sub_y_test))
            send_discord_message(f'== CIC Done Training: {dataset_name} with model: {name}, acc: {accuracy}, loss: {loss}, f1: {f1} ==')
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
# send_discord_message('== @everyone All training and evaluation is done ==')
print('== @everyone All training and evaluation is done ==')
