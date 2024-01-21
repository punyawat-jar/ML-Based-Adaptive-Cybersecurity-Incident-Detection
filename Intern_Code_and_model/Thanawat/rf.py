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
# List of file paths for training datasets
file_paths = []
for i in range (1,14):
    file = f"/home/s2316001/Clean_data/Clean_{i}-XX.csv"
    file_paths.append(file)
    
# Train multiple Random Forest models
num_models = len(file_paths)
for k in range(1,31):
    models = []
    for i, file_path in enumerate(file_paths):
        dataset = pd.read_csv(file_path)

        # Preprocess the dataset
        # Apply necessary preprocessing steps such as handling missing values, encoding categorical variables, etc.

        X = dataset.drop('Label', axis=1)  # Features
        X.drop(columns=X.columns[0], axis=1, inplace=True)
        y = dataset["Label"]  # Target variable

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        temp_1 = pd.concat([X_train,y_train],axis=1)
        temp_1.to_csv(f'/home/s2316001/Clean_data/{k}/train{i+1}.csv', encoding='utf-8')
        temp = pd.concat([X_val,y_val],axis=1)
        temp.to_csv(f'/home/s2316001/Clean_data/{k}/test{i+1}.csv', encoding='utf-8')
        model = RandomForestClassifier(n_estimators=100)  # You can adjust the hyperparameters as needed
        model.fit(X_train, y_train)
        filename = f'/home/s2316001/model_rfe/{k}/rfe_model{i+1}.sav'
        joblib.dump(model, filename)
        models.append(model)
        # Evaluate the model
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_train, y_pred)
        print(f"Model {i+1} Accuracy: {accuracy}")
        print(f"Model {i+1} precesion: {precision[1]}")
        print(f"Model {i+1} recall: {recall[1]}")
        print(f"Model {i+1} fbeta_score: {fbeta_score[1]}")
        matrix = metrics.confusion_matrix(y_train, y_pred)
        #print(matrix)
        # Plot Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()