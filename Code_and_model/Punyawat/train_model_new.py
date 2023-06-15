import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace "0" with the index of the desired GPU
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import joblib
import sys

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,  learning_curve

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# # Set the GPU device
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# np.set_printoptions(threshold=np.inf)
# # Configure TensorFlow to allocate memory on the GPU as needed
# physical_devices = tf.config.list_physical_devices("GPU")
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)

feature=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
          "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
          "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
          "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count", 
          "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
          "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

data_train = pd.read_csv('/home/s2316002/Project1/Dataset/KDDTrain+.txt',names = feature)
data_train.drop(['difficulty'],axis=1,inplace=True)

def Scaling(df_num, cols):
    log_df = np.log1p(df_num)  # Apply logarithmic scaling using np.log1p
    std_scaler = StandardScaler()
    std_scaler_temp = std_scaler.fit_transform(log_df)
    std_df = pd.DataFrame(std_scaler_temp, columns=cols)
    return std_df


cat_cols = ['is_host_login','protocol_type','service','flag','land', 'logged_in','is_guest_login', 'label']
def preprocess(dataframe):
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    scaled_df = Scaling(df_num, num_cols)
    
    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]
    
    dataframe.loc[dataframe['label'] == "normal", "label"] = 0
    dataframe.loc[dataframe['label'] != 0, "label"] = 1
    
    dataframe = pd.get_dummies(dataframe, columns=['protocol_type', 'service', 'flag'], dtype='int')
    return dataframe

data_train = preprocess(data_train)
train = data_train.drop(['label'],axis = 1)
test = data_train.label.astype('int')

text = '''
land wrong_fragment urgent hot num_failed_logins logged_in num_compromised root_shell num_outbound_cmds is_host_login is_guest_login count srv_count serror_rate srv_serror_rate rerror_rate srv_rerror_rate same_srv_rate diff_srv_rate srv_diff_host_rate dst_host_count dst_host_srv_count dst_host_same_srv_rate dst_host_diff_srv_rate dst_host_same_src_port_rate dst_host_srv_diff_host_rate dst_host_serror_rate dst_host_srv_serror_rate dst_host_rerror_rate dst_host_srv_rerror_rate protocol_type_icmp protocol_type_tcp protocol_type_udp service_IRC service_X11 service_Z39_50 service_auth service_bgp service_csnet_ns service_ctf service_discard service_domain service_echo service_eco_i service_ecr_i service_exec service_finger service_ftp service_ftp_data service_harvest service_hostnames service_http service_http_2784 service_http_443 service_http_8001 service_iso_tsap service_klogin service_kshell service_ldap service_login service_mtp service_name service_netbios_dgm service_netbios_ns service_netbios_ssn service_netstat service_nnsp service_ntp_u service_other service_pm_dump service_printer service_private service_red_i service_shell service_smtp service_sql_net service_ssh service_sunrpc service_supdup service_tftp_u service_tim_i service_time service_urh_i service_urp_i service_uucp service_uucp_path service_vmnet service_whois flag_OTH flag_REJ flag_RSTO flag_RSTOS0 flag_S0 flag_S1 flag_S2 flag_S3 flag_SF flag_SH
'''
selected_columns = text.split()

train = train[selected_columns]
time_steps = 50


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

X, y = create_dataset(train, test, time_steps)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# window_size = 50

# # # Reshape the input data to match the expected input shape of the model
# X_train = np.reshape(np.array(X_train), (X_train.shape[0], window_size, X_train.shape[1] // window_size))
# X_test = np.reshape(np.array(X_test), (X_test.shape[0], window_size, X_test.shape[1] // window_size))


model_shapes =  [(5, 2), (10, 3), (20, 4), (40, 5), (60, 6), (80, 7), (100, 8)]
dropout_rates = [0.01, 0.01, 0.15, 0.15, 0.20, 0.25, 0.30]
models = []

model_path = '/home/s2316002/Project1/DL_3/model/'
graph_path = '/home/s2316002/Project1/DL_3/graph/'

fig, axs = plt.subplots(2, 1, figsize=(8, 8))
for (neurons, layers), dropout_rate in zip(model_shapes, dropout_rates):
    # Create the model
    model = tf.keras.Sequential()
    for i in range(layers):
        if i == layers - 1:  # for the last LSTM layer
            model.add(tf.keras.layers.LSTM(neurons, return_sequences=False))
        else:  # for all other LSTM layers
            model.add(tf.keras.layers.LSTM(neurons, return_sequences=True))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))
    model._name = f'Hidden_layers_{neurons}'

    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])
    models.append(model)

for model in models:
    model.summary()

for model, (neurons, _) in zip(models, model_shapes):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=256)

    # Plot accuracy and loss
    axs[0].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[1].plot(history.history['loss'], label='Train Loss')
    axs[1].plot(history.history['val_loss'], label='Validation Loss')

    # Set labels and titles for the plots
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].set_title(f'Model Accuracy - Hidden Layers: {neurons}')

    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].set_title(f'Model Loss - Hidden Layers: {neurons}')

    # Save the figure
    fig.savefig(f'{graph_path}accuracy_loss_{neurons}.png')
    plt.cla()

    # Clear the current figure and create a new one for the next model
    plt.clf()
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # Save the model
    model.save(f'{model_path}model_{neurons}.h5')



print('=========================== Training complete ===========================')
