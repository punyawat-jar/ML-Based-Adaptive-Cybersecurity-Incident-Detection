import numpy as np
import requests, json
import os
import pandas as pd 
import glob
import gc
from sklearn.preprocessing import MinMaxScaler
os.chdir("C:\\Users\\Kotani Lab\\Desktop\\ML_senior_project\\ML-Based-Adaptive-Cybersecurity-Incident-Detection\\Code_and_model\\cic\\dataset")


directory = 'label_dataset'
if not os.path.exists(directory):
    os.makedirs(directory)
    
botnum=0
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

def preprocess(df):
    df.drop(['timestamp'], axis =1, inplace=True)
    df.loc[df['label'] == "BENIGN", "label"] = 0
    df.loc[df['label'] != 0, "label"] = 1

    scaler = MinMaxScaler()
    for col in df.columns:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).ravel()

    return df

def create_df(labels):
    saved_files = []

    # Loop over each label from the provided list and create separate datasets
    for i, label in enumerate(labels):
        send_discord_message(f'Starting {label} {i+1}/{len(labels)}')
        print(f'Starting {label} {i+1}/{len(labels)}')
        
        combined_df = df[df['label'].isin(['BENIGN', label])]
        filename = f".\\label_dataset\\{label}.csv"
        combined_df = preprocess(combined_df)
        combined_df.to_csv(filename, index=False)
        saved_files.append(filename)
        
        send_discord_message(f'Done {label}')
        print(f'Done {label}')

df = pd.read_csv('CIC_IDS2017.csv')
# df = df.drop(['timestamp'], axis =1)
labels = df.label.value_counts().index.tolist()

create_df(labels)
