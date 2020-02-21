import os 
import sys
print(sys.path)
from sd.models.unet import *
import torch
import torch.nn


ones = torch.ones(2,1,233,189)

import pdb; pdb.set_trace()
bu = BasicUNet(in_channels=1, out_channels=2, dimension=3)
print(bu.forward(ones).shape)

cu = UNet(n_channels=1, n_classes=2)
print(cu.forward(ones).shape)

print('ones: {}'.format(ones.shape))
