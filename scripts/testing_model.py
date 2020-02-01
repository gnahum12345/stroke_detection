import sys
import os 
print(sys.path)
sys.path.insert(0, os.path.abspath('./../'))

from unet import *
import torch
import torch.nn


ones = torch.ones(1,3,256,256)

import pdb; pdb.set_trace()
bu = BasicUNet()
bu.forward(ones)
