import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.losses import BinaryCrossentropy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
os.chdir('/home/s2316002/capstone_project/kdd/')
botnum = 1
bot = ['https://discord.com/api/webhooks/1162767976034996274/B6CjtQF1SzNRalG_csFx8-qJ5ODBoy5SBUelbGyl-v-QhYhwdsTfE59F-K-rXj3HyUh-',
      'https://discord.com/api/webhooks/1162767979658887299/0TICfekiC9wjPmp-GqE5zrwU57q2RJHG2peel_KOYagUDYCjovYUfyNJmDR9jbD-WXoE']

def processlabel(df):
    df.loc[df['label'] == 'normal', 'label'] = 0
    df.loc[df['label'] != 0, 'label'] = 1
    df['label'] = df['label'].astype('int')
    return df
    
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

def create_LSTM(n_input, n_features):
    model = Sequential()
    model.add(LSTM(256,return_sequences=True, input_shape =(n_input, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(256,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256,return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer ='adam', loss = 'BinaryCrossentropy')
    return model
    
df_list = glob.glob('./dataset/all_dataset/*.csv')
dataset = {}
for train_path in df_list:
    dir = train_path.split('/')
    print('Reading Data')
    df = pd.read_csv(f'./dataset/all_dataset/{dir[-1]}')
    df = processlabel(df)
    
    X = df.drop(['label'], axis =1)
    X = preprocess(X)
    y = df['label']

    print('Preprocess Data Completed')
    window_size = 128
    n_features = 41
    train_size = int(len(X) * 0.7)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    train_generator = TimeseriesGenerator(X_train, y_train, length = window_size, batch_size =8)
    train_generator = TimeseriesGenerator(X_test, y_test, length = window_size, batch_size =8)
    
    model = create_LSTM(window_size, n_features)
    
    print(model.summary())
    print('== Training ==')
    model.fit(generator, validation_data = (X_test, y_test))
