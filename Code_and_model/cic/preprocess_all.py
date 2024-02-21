import numpy as np
import pandas as pd 



def preprocess_normal(df, label):
    df.loc[df['label'] == "BENIGN", "label"] = 0
    df.loc[df['label'] == label, "label"] = 1
    df.loc[(df['label'] != 0) & (df['label'] != 1), "label"] = 0
    df['label'] = df['label'].astype('int')
    return df

def preprocess_mixed(df, label):
    df.loc[df['label'] == "BENIGN", "label"] = 0
    df.loc[df['label'] == label, "label"] = 1
    df.loc[(df['label'] != 0) & (df['label'] != 1), "label"] = 1
    df['label'] = df['label'].astype('int')
    return df

def main():
    
    df = pd.read_csv('./dataset/CIC_IDS2017.csv')
    
    labels = df.label.value_counts().index.tolist()
    
    for i, label in enumerate(labels):
        if label == 'BENIGN':
            print(f'SKIP {label}')
        else:
            print(f'Starting Normal {label} {i+1}/{len(labels)}')
            
            df = preprocess_normal(df.copy(), label)
            
            df.to_csv(f"./dataset/normal_dataset/{label}.csv", index=False)
            
    for i, label in enumerate(labels):
        if label == 'BENIGN':
            print(f'SKIP {label}')
        else:
            print(f'Starting Mixed {label} {i+1}/{len(labels)}')
            
            df = preprocess_mixed(df.copy(), label)
            
            df.to_csv(f"./dataset/mix_dataset/{label}.csv", index=False)

if __name__ == '__main__':
    main()
