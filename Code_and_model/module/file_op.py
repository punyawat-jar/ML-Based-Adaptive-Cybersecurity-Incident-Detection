import os

def makePath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_file(path):
    if os.path.isfile(path):
        print(f'{path} exist')
    else:
        raise Exception(f'Error: {path} not exist')
    
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