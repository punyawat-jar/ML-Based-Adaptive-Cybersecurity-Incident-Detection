import pandas as pd
import glob
import argparse


from module.testing_module import *

def main():
    parser = argparse.ArgumentParser(description='Weight reset code')
    
    parser.add_argument('--data',
                        dest='data_template',
                        type=str,
                        required=True,
                        help='The data struture. The default data structures is cic (CICIDS2017) and kdd (NSL-KDD). (*Require)')
    
    parser.add_argument('--changed-model',
                        dest='notdefault',
                        action='store_false',
                        help='The default weight loading. (*Require)')

    arg = parser.parse_args()
    data_template = arg.data_template
    notdefault = arg.notdefault

    weight_decimal = 3
    weight_path = f'{data_template}/weight.json'
    model_path = f'{data_template}/model.csv'
    threshold_path = f'{data_template}/Result/threshold.json'

    df_attacks = pd.read_csv(model_path)
    attack_list = df_attacks['attack'].tolist()

    df_train = pd.read_csv(glob.glob(f'{data_template}/train_test_folder/train_{data_template}/*')[0], skiprows=progress_bar())
    y_train = df_train['label']
    
    if notdefault == False:
        print(notdefault)
        y_train = y_train.apply(lambda x: x if x in attack_list else 'normal')

    og_attack_percent = read_attack_percent(y_train, weight_decimal)
    lowest_percent_attack = min(og_attack_percent, key=og_attack_percent.get)
    threshold = og_attack_percent[lowest_percent_attack]
    
    print('---- Resetting Weight and Threshold values ----')
    writingJson(og_attack_percent, weight_path)
    writingJson({'threshold': threshold}, threshold_path)
    print('---- Done Weight and Threshold values ----')

if __name__ == '__main__':
    main()