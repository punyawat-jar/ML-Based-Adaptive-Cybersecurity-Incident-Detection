import numpy as np
import requests, json
import os
import pandas as pd 
import multiprocessing
import gc
from sklearn.preprocessing import MinMaxScaler
os.chdir("C:\\Users\\Kotani Lab\\Desktop\\ML_senior_project\\ML-Based-Adaptive-Cybersecurity-Incident-Detection\\Code_and_model\\cic\\dataset")

directory = 'mix_dataset'
if not os.path.exists(directory):
    os.makedirs(directory)
    
botnum=1
bot = ['https://discord.com/api/webhooks/1120592724123467796/KLwL2pWifliFuwOzs_Az-VSGk8n1fz2lSEEEUsmEZ-UFgpRfWXHPmlZXW3AFhsmY4FWU',
      'https://discord.com/api/webhooks/1120592726648434838/qZejSK_KnJymtR8mN6uaXiouf1cK2GFgOTuPHbbd7TRGja3gOr4KcBAGxbJ_MAa93NP8',
      'https://discord.com/api/webhooks/1120594839344513055/kYPMXmHNqQaKACFfwSx-wsbeqawX8sfEanWAw4IMj-jQ6iSudhuxV17h6dC5BiCykW78',
      'https://discord.com/api/webhooks/1120595172204494888/HFqQW-_9a1zh5MR8_O8csVkKWlcUIxuB8jMd2daT6IPyDk_KgzDjFnQImcP5iPwa1KAz',
      'https://discord.com/api/webhooks/1132920901235654676/691VuY4nCL4yTjkqJHAtG6u3oXUxRonIxulHx3i3chJO92G0Ug6XxtdIyWVqyDk4vDLW',
      'https://discord.com/api/webhooks/1133199528284135515/uRBbJul9XFEA9YPqnvDSpZsQvSauZMzMdoBFnb8q69ILE_wVrxqxhkdDTeb-smGBmgIo']

def send_discord_message(content):
    webhook_url = bot[botnum]

    data = {
        'content': content
    }

    response = requests.post(webhook_url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

    if response.status_code != 204:
        raise ValueError(f'Request to discord returned an error {response.status_code}, the response is:\n{response.text}')
    

def label_preprocess(df, label):
    print(label)
    df.loc[df['label'] == "BENIGN", "label"] = 0
    df.loc[df['label'] == label, "label"] = 1
    df.loc[(df['label'] != 0) & (df['label'] != 1), "label"] = 2
    df['label'] = df['label'].astype('int')
    return df

def preprocess(df):
    
    df.loc[df['label'] == "BENIGN", "label"] = 0
    df.loc[df['label'] != 0, "label"] = 1

    scaler = MinMaxScaler()
    for col in df.columns:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).ravel()

    return df

df = pd.read_csv('Original_dataset-notuse\\CIC_IDS2017.csv')
df.drop(['timestamp'], axis =1, inplace=True)
labels = df.label.value_counts().index.tolist()

for i, label in enumerate(labels):
    if label == 'BENIGN' or label == 'DoS Hulk' or label == 'PortScan' or label == 'DDoS' or label == 'FTP-Patator' or label == 'DoS slowloris' or label == 'DoS Slowhttptest' or label == 'SSH-Patator' or label == 'Bot' or label == 'DoS GoldenEye':
        print(f'SKIP {label}')
    else:
        print(f'Starting {label} {i+1}/{len(labels)}')
        send_discord_message(f'Starting {label} {i+1}/{len(labels)}')
        df = label_preprocess(df.copy(), label)
        df = preprocess(df.copy())
        df.to_csv(f".\\{directory}\\{label}.csv", index=False)