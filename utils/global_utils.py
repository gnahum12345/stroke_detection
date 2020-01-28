import os
import gzip
import shutil

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def move_directories(old_location, new_directory):
    files = os.listdir(old_location)
    for file in files:
        path = os.path.join(old_location, file)
        if os.path.isdir(path):
            os.system('mv {} {}'.format(path, new_directory))
            print(bcolors.OKBLUE + 'Moved {} to new location {}'.format(file, new_directory) + bcolors.ENDC)
    print(bcolors.HEADER + 'Moved all directories' + bcolors.ENDC)

def unzip_data(home_directory, remove_file=False):
    for root, dirs, files in os.walk(home_directory):
        print(root, dirs, files)
        for file in files:
            if '.gz' in file:
                unzip_filename = file.strip('.gz')
                path = os.path.join(root, file)
                new_path = os.path.join(root, unzip_filename)

                f_in = gzip.open(path, 'rb')
                f_out = open(new_path, 'wb')
                shutil.copyfileobj(f_in, f_out)
                f_in.close()
                f_out.close()
                if remove_file:
                    os.remove(path)
                    print(bcolors.FAIL + 'Filed Removed' + bcolors.ENDC)

        print(root)
        print(bcolors.OKGREEN + '%'*5 + ' FINISHED WITH ' + root + ' ' + '%'*5 + bcolors.ENDC)

    print(bcolors.OKBLUE + 'All files have been unziped.' + bcolors.ENDC)
