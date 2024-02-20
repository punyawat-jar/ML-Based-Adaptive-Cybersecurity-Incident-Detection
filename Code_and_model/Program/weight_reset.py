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
    
    arg = parser.parse_args()
    data_template = arg.data_template
    
    weight_decimal = 3
    weight_path = f'{data_template}/weight.json'
    threshold_path = f'{data_template}/Result/threshold.json'
    
    
    df_train = pd.read_csv(glob.glob(f'{data_template}/train_test_folder/train_{data_template}/*')[0], skiprows=progress_bar())
    y_train = df_train['label']
    og_attack_percent = read_attack_percent(y_train, weight_decimal)
    # print(og_attack_percent)
    lowest_percent_attack = min(og_attack_percent, key=og_attack_percent.get)
    threshold = og_attack_percent[lowest_percent_attack]
    
    writingJson(og_attack_percent, weight_path)
    writingJson({'threshold': threshold}, threshold_path)
    
if __name__ == '__main__':
    main()