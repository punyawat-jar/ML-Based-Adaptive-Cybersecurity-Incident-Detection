import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier 
import joblib
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py
import joblib
from sklearn import model_selection, feature_selection, utils, ensemble, linear_model, metrics, kernel_approximation
import random
import seaborn as sns
nb_prediction = 30
accuracy_new = np.array([0.]*nb_prediction)
precision_new = np.array([0.]*nb_prediction)
recall_new = np.array([0.]*nb_prediction)
fbeta_new = np.array([0.]*nb_prediction)
for i in range (1,14):
    print(f'-------------- Model {i} -------------------')
    for z in range (1,14):
        for k in range(0,30):   
            print(f'-------------round {k}-------------------')
            model_path = f'/home/s2316001/model_rfe/{k+1}/rfe_model{i}.sav'
            model = joblib.load(model_path)
            df = pd.read_csv(f"/home/s2316001/Clean_data/{k+1}/test{z}.csv")
            X = df.drop(columns=[df.columns[0],"Label"], axis=1)
            y = df['Label']
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y, y_pred)
            print(f'-------------- file {z} -------------------')
            print(f"Model {i} Accuracy: {accuracy}")
            print(f"Model {i} precesion: {precision[1]}")
            print(f"Model {i} recall: {recall[1]}")
            print(f"Model {i} fbeta_score: {fbeta_score[1]}")
            accuracy_new[k] = accuracy
            precision_new[k] = precision[1]
            recall_new[k] = recall[1]
            fbeta_new[k] = fbeta_score[1]
            matrix = metrics.confusion_matrix(y, y_pred)
            #print(matrix)
            # Plot Confusion Matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.savefig(f'/home/s2316001/Ensemble_matrix/{k+1}/file_manual{z}.png')
            plt.show()
        print(f"-----------------Model{i} test{z}---------------")
        print("Test")
        print("Accuracy:",accuracy_new.mean(), accuracy_new.std(), accuracy_new)
        print("precision = ", precision_new.mean(), precision_new.std(), precision_new)
        print("recall = ", recall_new.mean(), recall_new.std(), recall_new)
        print("fbeta_score = ", fbeta_new.mean(), fbeta_new.std(), fbeta_new)
            
         