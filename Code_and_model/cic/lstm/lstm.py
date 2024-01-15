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

import gc
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.losses import BinaryCrossentropy
from keras.utils import to_categorical


from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss



gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
else:
    strategy = tf.distribute.OneDeviceStrategy("GPU:0")
    print('Single device: GPU:0')


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")



# Create input sequences using sliding windows

def data_generator(df, indices, window_size, sliding_window):
    for idx in indices:
        if idx + window_size > len(df):
            continue
        sequence = df.iloc[idx:idx + window_size, :-1].values
        label = df.iloc[idx + window_size - 1, -1]
        yield sequence, label



def create_LSTM(window_size, n_features):
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=512, activation='tanh', input_shape=(window_size, n_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=512, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=256, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model
    optimizer = RMSprop(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model
    
def process_chunk(df_chunk):
    df_chunk['label'] = label_encoder.transform(df_chunk['label'])
    return df_chunk

botnum = 0
bot = ['https://discord.com/api/webhooks/1162767976034996274/B6CjtQF1SzNRalG_csFx8-qJ5ODBoy5SBUelbGyl-v-QhYhwdsTfE59F-K-rXj3HyUh-',
      'https://discord.com/api/webhooks/1162767979658887299/0TICfekiC9wjPmp-GqE5zrwU57q2RJHG2peel_KOYagUDYCjovYUfyNJmDR9jbD-WXoE']

def send_discord_message(content):
    webhook_url = bot[botnum]

    data = {
        'content': content
    }

    response = requests.post(webhook_url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

    if response.status_code != 204:
        raise ValueError(f'Request to discord returned an error {response.status_code}, the response is:\n{response.text}')

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
                message = f"LSTM-CIC -> Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}"
                payload = {"content": message}
                headers = {"Content-Type": "application/json"}
                response = requests.post(self.webhook_url, data=json.dumps(payload), headers=headers)


os.chdir("C:\\Users\\Kotani Lab\\Desktop\\ML_senior_project\\ML-Based-Adaptive-Cybersecurity-Incident-Detection\\Code_and_model\\cic")

df_list = glob.glob('.\\dataset\\mix_dataset\\*.csv')
# dataset = {}

model_path = '.\\lstm\\model'
csv_path = '.\\lstm\\results'


create_directory(model_path)
create_directory(csv_path)

label_encoder = LabelEncoder()


sliding_window = 1
window_size = 128
batch_size = 1024
#Training and evaluation
print('Starting')
print(df_list)


for train_path in df_list:
    
    dir = train_path.split('\\')
    file_name = dir[-1]
    print(f'Reading Dataset: {file_name}')

    df = pd.read_csv(f'.\\dataset\\mix_dataset\\{file_name}')
    print('Preprocessing')
    print('Label Encoder')
    df['label'] = label_encoder.fit_transform(df['label'])
    df['label'] = df['label'].replace(2, 0)
    indices = np.arange(len(df) - window_size + 1)
    train_indices, test_indices = train_test_split(indices, test_size=0.15, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.17647, random_state=42)        ## 70 / 30 train and val+test
    
    steps_per_epoch = len(train_indices) // batch_size
    validation_steps=len(val_indices) // batch_size

    print('Creating Dataset')
    # Training dataset
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(df, train_indices, window_size, sliding_window),
        output_types=(tf.float32, tf.float32),
        output_shapes=((window_size, df.shape[1] - 1), ())
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Validation dataset
    val_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(df, val_indices, window_size, sliding_window),
        output_types=(tf.float32, tf.float32),
        output_shapes=((window_size, df.shape[1] - 1), ())
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Testing dataset
    test_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(df, test_indices, window_size, sliding_window),
        output_types=(tf.float32, tf.float32),
        output_shapes=((window_size, df.shape[1] - 1), ())
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)


    print('Creating Callbacks & Setting variable')
    #Creating callbacks
    save_model_path = f'.\\lstm\\model\\{file_name}.ckpt'

    discord_callback = DiscordNotificationCallback(bot[botnum], interval=1)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='min')
    best_model = ModelCheckpoint(save_model_path, save_best_only=True, monitor='val_loss', mode='min')



    #Setting parameter

    n_features = df.shape[-1] -1


    model = create_LSTM(window_size, n_features)
    print('Model Summary')
    #Summary of Model
    model.summary()
    try:
        print('Training model')
        history = model.fit(
            train_dataset,
            epochs=10,
            steps_per_epoch=steps_per_epoch,
            callbacks=[discord_callback, best_model, earlyStopping],
            validation_data=val_dataset,
            validation_steps=validation_steps
        )

        model.load_weights(save_model_path)

        y_pred_prob = model.predict(test_dataset)
        y_pred_test = (y_pred_prob > 0.5).astype(int)
        y_test_true = np.concatenate([y for _, y in test_dataset], axis=0)

        accuracy_test = accuracy_score(y_test_true, y_pred_test)
        f1_test = f1_score(y_test_true, y_pred_test)
        precision_test = precision_score(y_test_true, y_pred_test)
        recall_test = recall_score(y_test_true, y_pred_test)

        # loss_test = log_loss(y_test_true, y_pred_prob)

        # Save results
        results = {
            "accuracy_test": accuracy_test,
            "f1_test": f1_test,
            "precision_test": precision_test,
            "recall_test": recall_test,
            
        }


        results_df = pd.DataFrame([results], columns=results.keys())
        results_df.to_csv(f".\\lstm\\results\\{file_name}.csv", index=False)
        shutil.move(f'.\\dataset\\mix_dataset\\{file_name}',
        f'.\\dataset\\mix_done\\{file_name}')
        print(results_df)
    except Exception as error:
        print(f'Error : {error}')
        
    del df
    gc.collect()