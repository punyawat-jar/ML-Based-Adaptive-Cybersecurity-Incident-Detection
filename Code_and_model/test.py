import os
import argparse



def main():
    parser = argparse.ArgumentParser(description='Testing code')
    parser.add_argument('--data',
                        dest='data_template',
                        type=str,
                        required=True,
                        help='The data struture. The default data structures is cic (CICIDS2017) and kdd (NSL-KDD). (*Require)')
    


    parser.add_argument('--weight',
                        dest='weight_csv',
                        type=str,
                        help="The csv file's location for inputing an integration's weights for models.")
    
    parser.add_argument('--model',
                        dest='model_loc',
                        type=str,
                        help='The trained models loaction.')

    parser.add_argument('--sequence',
                        dest='sequence',
                        type=bool,
                        default=True,
                        help='The sequence mode show the network in sequence with the prediction of each attack')
    
    parser.add_argument('--debug',
                        dest='debug',
                        type=bool,
                        default=False,
                        help='The debug model show the list of the prediction categories that might be an attack of the model.')

    arg = parser.parse_args()

    data_template = arg.data_template

    weight_csv = arg.weight_csv if arg.weight_csv is not None else f'./{data_template}/weight.csv'

    model_loc =  arg.model_loc if arg.model_loc is not None else f'./{data_template}/model'

    
    print(os.getcwd())
    



if __name__ == '__main__':
    main()