import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import shutil
import requests, json

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss

from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

botnum = 1
bot = ['https://discord.com/api/webhooks/1162767976034996274/B6CjtQF1SzNRalG_csFx8-qJ5ODBoy5SBUelbGyl-v-QhYhwdsTfE59F-K-rXj3HyUh-',
      'https://discord.com/api/webhooks/1162767979658887299/0TICfekiC9wjPmp-GqE5zrwU57q2RJHG2peel_KOYagUDYCjovYUfyNJmDR9jbD-WXoE']

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
else:
    strategy = tf.distribute.OneDeviceStrategy("GPU:0")
    print('Single device: GPU:0')


class DiscordNotificationCallback(Callback):
    def __init__(self, webhook_url, interval=1):
        super().__init__()
        self.webhook_url = webhook_url
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            if logs is not None:
                loss = logs.get('loss')
                accuracy = logs.get('accuracy')
                val_loss = logs.get('val_loss')
                val_accuracy = logs.get('val_accuracy')
                message = f"LSTM-KDD -> Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}"
                payload = {"content": message}
                headers = {"Content-Type": "application/json"}
                response = requests.post(self.webhook_url, data=json.dumps(payload), headers=headers)

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
    model.add(LSTM(1,return_sequences=False, input_shape =(n_input, n_features)))
    model.add(Dropout(0.2))
    # model.add(LSTM(256,return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(256,return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer ='adam', loss = 'BinaryCrossentropy', metrics=['accuracy'])
    return model

os.chdir('/home/s2316002/ML-Based-Adaptive-Cybersecurity-Incident-Detection/Code_and_model/kdd/')
df_list = glob.glob('/home/s2316002/ML-Based-Adaptive-Cybersecurity-Incident-Detection/Code_and_model/kdd/dataset/all_dataset/*.csv')
dataset = {}

model_path = './lstm/model'
csv_path = './lstm/results'
if not os.path.exists(model_path):

   # Create a new directory because it does not exist
   os.makedirs(model_path)
   print(f"The {model_path} directory is created!")

if not os.path.exists(csv_path):

   # Create a new directory because it does not exist
   os.makedirs(csv_path)
   print(f"The {csv_path} directory is created!")


#Training and evaluation

for train_path in df_list:

   #Loading Dataset & Preprocess
   dir = train_path.split('/')
   file_name = dir[-1]
   df = pd.read_csv(f'./dataset/all_dataset/{file_name}')
   df = processlabel(df)
    
   X = df.drop(['label'], axis =1)
   X = preprocess(X)
   y = df['label']

   #Creating callbacks
   save_model_path = f'./lstm/model/{file_name}.ckpt'

   discord_callback = DiscordNotificationCallback(bot[botnum], interval=1)
   earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='min')
   best_model = ModelCheckpoint(save_model_path, save_best_only=True, monitor='val_loss', mode='min')
   if isinstance(X, (pd.DataFrame, pd.Series)):
      X = X.values

   if isinstance(y, (pd.DataFrame, pd.Series)):
      y = y.values

   #Setting parameter
   window_size = 128
   n_features = 41
   train_size = int(len(X) * 0.7)
    
   X_train, X_test = X[:train_size], X[train_size:]
   y_train, y_test = y[:train_size], y[train_size:]
   
   train_generator = TimeseriesGenerator(X_train, y_train, length = window_size, batch_size = 8)
   test_generator = TimeseriesGenerator(X_test, y_test, length = window_size, batch_size=len(X_test)-window_size)
    
   model = create_LSTM(window_size, n_features)

   #Summary of Model
   model.summary()
   try:
    model.fit(train_generator, 
                epochs=1,
                callbacks= [discord_callback, best_model, earlyStopping],
                validation_data=test_generator)  
    
    evaluation_model = create_LSTM(window_size, n_features)
    evaluation_model.load_weights(save_model_path)

    # Evaluate the model using the test generator sequences
    X_test_sequences, y_test_sequences = test_generator[0]
    y_pred_prob_test = evaluation_model.predict(test_generator)
    y_pred_test = (y_pred_prob_test > 0.5).astype(int)

    accuracy_test = accuracy_score(y_test_sequences, y_pred_test)
    f1_test = f1_score(y_test_sequences, y_pred_test)
    precision_test = precision_score(y_test_sequences, y_pred_test)
    recall_test = recall_score(y_test_sequences, y_pred_test)
    loss_test = log_loss(y_test_sequences, y_pred_prob_test)

    # Save results
    results = {
        "accuracy_test": accuracy_test,
        "f1_test": f1_test,
        "precision_test": precision_test,
        "recall_test": recall_test,
        "loss_test": loss_test
    }


    results_df = pd.DataFrame([results], columns=results.keys())
    results_df.to_csv(f"./lstm/results/{file_name}.csv", index=False)

    print(results_df)
   except Exception as error:
    print(f'Error : {error}')