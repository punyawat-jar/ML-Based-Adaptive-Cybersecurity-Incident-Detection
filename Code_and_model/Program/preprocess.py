import os
import pandas as pd
import numpy as np
import argparse
import glob
import sys
import traceback

from multiprocessing import cpu_count

from module.file_op import *
from module.preprocess_KDD import ProcessKDD
from module.preprocess_CIC import ProcessCIC
from module.file_converter import *
from module.util import *

def main():
    try:
        parser = argparse.ArgumentParser(description='Testing code')
        parser.add_argument('--data',
                            dest='data_template',
                            type=str,
                            required=True,
                            help='The data struture. The default data structures is cic (CICIDS2017) and kdd (NSL-KDD). (*Require)')

        parser.add_argument('--model',
                            dest='model_loc',
                            type=str,
                            help='The trained models loaction.')
        
        parser.add_argument('--network',
                            dest='net_file_loc',
                            type=str,
                            help='The netowrk file location (.csv)')

        parser.add_argument('--input_dataset',
                            dest='input_dataset',
                            type=str,
                            help='input_dataset CIC-IDS2017.csv or KDD.csv, the concat files that already processed.')
        
        parser.add_argument('--usingMultiprocess',
                            dest='multiCPU',
                            action=argparse.BooleanOptionalAction,
                            help='multiCPU is for using all the process.')
        
        parser.add_argument('--n_Process',
                            dest='num_processes',
                            type=str,
                            help='num_processes is the number of process by user, default setting is all process (cpu_count()).')
        
        
        arg = parser.parse_args()

        data_template = arg.data_template

        net_file_loc = arg.net_file_loc if arg.net_file_loc is not None else f'./{data_template}/dataset/InputDataset/'
        
        input_dataset = arg.input_dataset 
        
        multiCPU = arg.multiCPU
        
        num_processes = int(arg.num_processes) if arg.num_processes is not None else cpu_count()
        
        #File path
        os.chdir('./Code_and_model/Program') ##Change Working Directory
        print(os.getcwd())
        file_path = glob.glob(net_file_loc+'/*', recursive=True)
        print(f'Data Path : {net_file_loc}')
        print(f'file_path : {file_path}')
        
        if not file_path:
            raise Exception('The dataset path is contain no files')

        check_data_template(data_template)
        

        
        makePath(f'./{data_template}')
        makePath(f'./{data_template}/dataset')
        makePath(f'./{data_template}/dataset/mix_dataset')
        
        if data_template == 'kdd':
            ProcessKDD(file_path, input_dataset, multiCPU, num_processes)

        elif data_template == 'cic':
            ProcessCIC(file_path, input_dataset, multiCPU, num_processes)
            
    except Exception as E:
        print(E)
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()