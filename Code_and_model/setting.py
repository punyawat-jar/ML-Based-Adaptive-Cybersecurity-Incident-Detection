import os
import argparse
import glob

from module.file_op import *
from module.util import *

#The setting.py is to setting the environment.

def main():
    
    parser = argparse.ArgumentParser(description='Testing code')
    parser.add_argument('--data',
                        dest='data_template',
                        type=str,
                        required=True,
                        help='The data struture. The default data structures is cic (CICIDS2017) and kdd (NSL-KDD). (*Require)')
    
    arg = parser.parse_args()
    data_template = arg.data_template
    
    check_data_template(data_template)
    
    train_test_folder = [f'.{data_template}/train_test_folder/train_{data_template}',
                         f'.{data_template}/train_test_folder/test_{data_template}']
    result_path = f'.{data_template}/Result'
    
    training_result_path = [f'{data_template}/Training/confusion_martix',
                            f'{data_template}/Training/model',
                            f'{data_template}/Training/compare']
    makePath(train_test_folder)
    
    mix_directory = 'mix_dataset'
    makePath(f'./{data_template}')
    makePath(f'./{data_template}/dataset')
    makePath(f'./{data_template}/{mix_directory}')
    makePath(result_path)
    
if __name__ == '__main__':
    main()