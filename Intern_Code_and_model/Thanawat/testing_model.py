# stacked generalization with linear meta model on blobs dataset
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.utils import to_categorical
from numpy import dstack
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import seaborn as sns
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py
import joblib
from tensorflow.keras.models import load_model
from sklearn import model_selection, feature_selection, kernel_approximation, ensemble, linear_model, metrics, utils

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
# load models from file
def load_all_models(n_models):
    all_models = list()
    for i in range(1,n_models):
     # define filename for this ensemble
        filename = f'/home/s2316001/codepaper/LSTM/lstm_model{i}.h5'
     # load model from file
        model = load_model(filename)
     # add to list of members
        all_models.append(model)
        #print('>loaded %s' % filename)
    return all_models
 
# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
     # make prediction
        yhat = model.predict(inputX, verbose=0)
     # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
     # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX
 
# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
    
     # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
     # fit standalone model
    model = LogisticRegression()
    model.fit(stackedX, inputy)
    return model
 
# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
 # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
 # make a prediction
    yhat = model.predict(stackedX)
    return yhat
# X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# load all models
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py
import pickle
from sklearn import model_selection, feature_selection, linear_model, metrics
n_members = 14
members = load_all_models(n_members)
#print("Import data")
for j in range (7,14):
    X = pd.read_hdf(f'/home/s2316001/codepaper/data_window_botnet{j}.h5', key='data')
    X.reset_index(drop=True, inplace=True)

    X2 = pd.read_hdf(f'/home/s2316001/codepaper/data_window3_botnet{j}.h5', key='data')
    X2.reset_index(drop=True, inplace=True)

    X = X.join(X2)

    X.drop('window_id', axis=1, inplace=True)

    y = X['Label_<lambda>']
    X.drop('Label_<lambda>', axis=1, inplace=True)

    labels = np.load(f"/home/s2316001/codepaper/data_window_botnet{j}_labels.npy",allow_pickle=True)

    #print(X)
    #print(y)
    # print(X.columns.values)
    # print(labels)

    y_bin6 = y==np.where(labels == 'flow=From-Botne')[0][0]
    #X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_bin6, test_size=0.33)
    #y_train_bin6 = y_train==6
    #y_test_bin6 = y_test==6

    #print("y", np.unique(y, return_counts=True))
    #X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    nb_prediction = 50
    accuracy_new = np.array([0.]*nb_prediction)
    precision_new = np.array([0.]*nb_prediction)
    recall_new = np.array([0.]*nb_prediction)
    fbeta_new = np.array([0.]*nb_prediction)
    for i in range(0,nb_prediction):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_bin6, test_size=0.33, random_state=None)
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    #print('Loaded %d models' % len(members))
    # evaluate standalone models on test dataset
    # for model in members:
    #     _, acc = model.evaluate(X_test, y_test, verbose=0)
    #     print('Model Accuracy: %.3f' % acc)
    # fit stacked model using the ensemble
        model = fit_stacked_model(members, X_test, y_test)
    # evaluate model on test set
        yhat = stacked_prediction(members, model, X_test)
        accuracy_new[i] = accuracy_score(y_test , yhat)
        precision_new[i] = precision_score(y_test , yhat)
        recall_new[i] = recall_score(y_test , yhat)
        fbeta_new[i] = f1_score(y_test , yhat)
        matrix = metrics.confusion_matrix(y_test, yhat)
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(f'/home/s2316001/codepaper/model_LSTM_ensemble_test/LSTM_ensemble_confusion_matrix_file{j}_No{i}.png')
        plt.close()
        filename = f'/home/s2316001/codepaper/LSTM_Ensemble/Ensemble_lstm_file{j}_model{i}.h5'
        joblib.dump(model, filename)
        print(f'---------------done {i}-------------')
    print(f"-----------------Total {j}---------------")
    print("Test")
    print("Accuracy:",accuracy_new.mean(), accuracy_new.std(), accuracy_new)
    print("precision = ", precision_new.mean(), precision_new.std(), precision_new)
    print("recall = ", recall_new.mean(), recall_new.std(), recall_new)
    print("fbeta_score = ", fbeta_new.mean(), fbeta_new.std(), fbeta_new)
joblib.dump(model, f'/home/s2316001/codepaper/LSTM_Ensemble/Last_Ensemble_lstm_model.h5')