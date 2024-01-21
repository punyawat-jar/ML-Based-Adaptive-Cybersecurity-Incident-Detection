import numpy as np
import requests, json
import os
import pandas as pd 

os.chdir("C:\\Users\\Kotani Lab\\Desktop\\ML_senior_project\\ML-Based-Adaptive-Cybersecurity-Incident-Detection\\Code_and_model\\kdd\\dataset")

def create_df(labels):
    saved_files = []

    # Loop over each label from the provided list and create separate datasets
    for i, label in enumerate(labels):
        send_discord_message(f'Starting {label} {i+1}/{len(labels)}')
        print(f'Starting {label} {i+1}/{len(labels)}')
        
        combined_df = df[df['label'].isin(['normal', label])]
        filename = f"./{directory}/{label}.csv"
        combined_df.to_csv(filename, index=False)
        saved_files.append(filename)
        
        # send_discord_message(f'Done {label}')
        print(f'Done {label}')
        
df = pd.read_csv('all+.csv')
labels = df.label.value_counts().index.tolist()

create_df(labels)
send_discord_message(f'== ALL Done ==')