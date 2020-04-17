import psutil
import time

def checkProcessRunning(pid): 
    for proc in psutil.process_iter(): 
        if proc.pid == pid:
            return True
    
    return False 

process = int(input('pid: '))
script = input('name of script: ')
while checkProcessRunning(process): 
    time.sleep(20)

import os
os.system('. %s' % script)
