import os
import pandas as pd
import numpy as np
import argparse
import glob
import sys
import traceback

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

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
                            required=True,
                            help='The netowrk file location (.csv)')

        parser.add_argument('--input_dataset',
                            dest='input_dataset',
                            type=str,
                            help='input_dataset CIC-IDS2017.csv or KDD.csv, the concat files that already processed.')
        
        arg = parser.parse_args()

        data_template = arg.data_template

        model_loc =  arg.model_loc if arg.model_loc is not None else f'./{data_template}/model'

        net_file_loc = arg.net_file_loc
        
        input_dataset = arg.input_dataset
        
        
        #File path
        os.chdir('./Code_and_model') ##Change Working Directory
        
        file_path = glob.glob(net_file_loc+'/*', recursive=True)
        file_type = file_path[0].split('.')[-1]
        mix_directory = 'mix_dataset'
        
        check_data_template(data_template)
        

        
        makePath(f'./{data_template}')
        makePath(f'./{data_template}/dataset')
        makePath(f'./{data_template}/{mix_directory}')
        
        if data_template == 'kdd':
            ProcessKDD(file_path, mix_directory, input_dataset)

        elif data_template == 'cic':
            ProcessCIC(file_path, mix_directory, input_dataset)
            
    except Exception as E:
        print(E)
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()