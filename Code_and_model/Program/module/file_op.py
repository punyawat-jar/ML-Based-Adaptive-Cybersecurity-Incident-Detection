import os
import pandas as pd
import json

def makePath(path):
    if isinstance(path, list):
        for ipath in path:
            if not os.path.exists(ipath):
                print(f'Creating : {ipath}')
                os.makedirs(ipath, exist_ok=True)
    else:
        if not os.path.exists(path):
            print(f'Creating : {path}')
            os.makedirs(path, exist_ok=True)

def check_file(path):
    if os.path.isfile(path):
        print(f'{path} exist')
    else:
        raise Exception(f'Error: {path} not exist')

def check_and_return_file(path):
    print('Checking file...')
    if os.path.isfile(path):
        print(f'File exist at {path}')
        return True
    else:
        print(f'File not exist, creating file at {path}')
        return False
        
def checkFileName(paths):
    path_list = []
    for path in paths:
        path_list.append(path.split('.')[-1])
    
    first_item = path_list[0]
    
    for item in path_list:
        if first_item != item:
            return False, first_item
    return True, first_item

def list_of_file_contain(text, list):
    return [filename for filename in list if text in filename.lower()]

def writingJson(data, path):
    with open(path, 'w') as file:
        json.dump(data, file)
    print(f'Writing Json : {path}')
