import os 
import sys
print(sys.path)
from sd.models.unet import *
import torch
import torch.nn


ones = torch.ones(2,192,233,189)

import pdb; pdb.set_trace()
bu = BasicUNet(in_channels=192, out_channels=2)
bu.forward(ones)
