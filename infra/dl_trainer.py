import argparse
import os 
import numpy as np 
import torch 
import torch.nn as nn
from torch import optim 
from tqdm import tqdm 

from torch.utils.data import DataLoader, random_split
from sd.infra.logger import Logger
from sd.infra.dataset import AtlasDataset
from sd.models.unet import * 
from sd.infra.global_utils import *
    

class DL_Trainer(object): 

    def __init__(self, params): 
        '''
            params should contain the following arguments: 
            Keys: 
               - dataset
               - device
               - epoches
               - frequency
               - learning rate (lr)
               - logger 
               - optimizer (adam/sdg/agd)
               - seed 
        ''' 

        # Storing class variables. 
        self.params = params 
        self.logger = params.logger
        self.device = params.device
        self.model = params.model 
        self.dataset = params.dataset   
        self.train, self.val = self.split(self.dataset) # todo (gn): implement this function
        self.optimizer = params.optimizer
        self.freq = params.freq
        self.seed = params.seed 
        self.loss = params.loss 
        # Setting random seed for reproducibility 
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


    def run_training_loop(self, n_iter, n_epoch): 
        for i in range(n_iter): 
            for epoch in range(n_epoch): 
                
