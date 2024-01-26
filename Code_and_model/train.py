import os
import traceback
import argparse
import sys

def main():
    try:
        parser = argparse.ArgumentParser(description='Training code')
        parser.add_argument('--')
        
        
        arg = parser.parse_args()
        
        
        #File path
        os.chdir('./Code_and_model') ##Change Working Directory
        
        
    except Exception as E:
        print(E)
        traceback.print_exc()
        sys.exit(1)



if __name__ == '__main__':
    main()