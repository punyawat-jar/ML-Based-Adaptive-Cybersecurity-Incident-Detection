from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

def getModel():
    models = {
        # 'LogisticRegression': LogisticRegression(max_iter=10000),
        # 'ExtraTrees': ExtraTreesClassifier(),
        # 'LDA': LinearDiscriminantAnalysis(),
        # 'QDA': QuadraticDiscriminantAnalysis(),
        # 'DecisionTree': DecisionTreeClassifier(),
        # 'RandomForest': RandomForestClassifier(),
        # 'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), n_jobs=-1),
        # 'GradientBoosting': GradientBoostingClassifier(),
        # 'KNeighbors': KNeighborsClassifier(),
        # 'GaussianNB': GaussianNB(),
        # 'Perceptron': Perceptron(),
        # 'AdaBoost': AdaBoostClassifier()
    }
    
    return models



def sequential_models(window_size, n_features):

    models = {
        # 'LSTM' : lstm(window_size, n_features)
    }

    return models

def lstm(window_size, n_features):
    model = Sequential(name='LSTM')
    model.add(LSTM(512, input_shape=(window_size, n_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    optimizer = RMSprop(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model