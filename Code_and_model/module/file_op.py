import os

def makePath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_file(path):
    if os.path.isfile(path):
        print(f'{path} exist')
    else:
        raise Exception(f'Error: {path} not exist')
    
def checkFileName(path):
    if path.endswith('.txt'):
        return 'txt'
    elif path.endswith('.csv'):
        return 'csv'
    elif path.endswith('.json'):
        return '.json'
    else:
        return 'unsupported'
    
