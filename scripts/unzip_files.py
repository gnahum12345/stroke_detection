import sys
sys.path.insert(1, '../utils')
import os
import gzip
import shutil

from global_utils import *


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--home_directory', '-hd', '-dir', type=str, default=None)
    parser.add_argument('--remove_file', '-rm', action='store_true')
    args = parser.parse_args()
    params = vars(args)
    if params['home_directory'] == None:
        home_directory = input('which folder would you like to unzip all files? \n')
    else:
        home_directory = params['home_directory']
    remove_file = params['remove_file']
    unzip_data(home_directory,remove_file)

if __name__ == '__main__':
    main()
