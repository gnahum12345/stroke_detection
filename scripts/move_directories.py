import os
from global_utils import *



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--home_directory', '-hd', '-dir', type=str, default=None)
    parser.add_argument('--new_directory', '-nd', '--new_dir', type=str, default=None)
    args = parser.parse_args()
    params = vars(args)
    if params['home_directory'] == None:
        home_directory = input('which folder would you like to move all the directories from? \n')
    else:
        home_directory = params['home_directory']
    if params['new_directory'] == None:
        new_directory = input('which folder would you like to move all directories to? \n')
    else:
        new_directory = params['new_directory']

    move_directories(home_directory, new_directory)
