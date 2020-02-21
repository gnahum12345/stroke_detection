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
        self.net = params.model
        self.dataset = params.dataset
        self.optimizer_type = params.optimizer_type
        self.lr = params.lr
        self.freq = params.freq
        self.seed = params.seed
        self.optimizer = self.get_optimizer() # requires: model, lr, optimizer_type
        self.batch_size = params.batch_size
        # setting criterion
        self.criterion = params.loss_fn
        # Setting random seed for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.cuda = False 
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True 
            self.cuda = True 
            self.net = self.net.type(torch.cuda.DoubleTensor)


        self.use_volumes = self.net.dimension == 3
        self.train, self.val, self.test = self.split(self.dataset, 0.7,0.2)
        
        
        self.global_step = 0
        self.logger.log('dl_trainer', 'Finished setting up dl_trainer. \n %s' % self._get_description())

    def _get_description(self):
        sb = 'params: {0}\nlogger{1}\ndevice {2}\nnet{3}\ndataset {4}\ncriterion {5}\noptimizer {6}'
        return sb.format(str(self.params), str(self.logger), str(self.device), str(self.net), str(self.dataset), str(self.criterion), str(self.optimizer))

    def get_optimizer(self):
        name_opt = self.optimizer_type.strip().lower()
        if name_opt == 'adam':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        elif name_opt == 'sgd': 
            optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

        self.logger.log('Optimizer', 'Optimizer set as {0}, \n {1}'.format(self.optimizer_type, str(optimizer)))
        return optimizer

    def split(self, data, train, valid):
        '''
            Splits the data into train, validation and testing.
            Note that ```train + valid = 1 - test.```
        '''
        assert train + valid <= 1, 'Train + Valid <= 1'
        n = len(data)
        train_s = int(n*train)
        valid_s = int(n*valid)
        test_s = n - train_s - valid_s
        return random_split(data, [train_s, valid_s, test_s])


    def process_slices(self, mask, scan, pbar): 
        scan_dl = DataLoader(scan, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        mask_dl = DataLoader(mask, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        epoch_loss = 0 
        for scan_slice, true_mask in zip(scan_dl, mask_dl):
            if (self.cuda): 
                scan_slice = scan_slice.cuda()
                true_mask = true_mask.cuda()

            loss = self.process_object(true_mask, scan_slice)
            
            epoch_loss += loss.item()
            self.logger.scalar_summary('loss/train', loss.item(), self.global_step)
            self.logger.scalar_summary('cum_loss/train', epoch_loss, self.global_step)
            pbar.set_postfix(**{f'loss(batch: {self.batch_size})': loss.item(), f'Average loss (across {self.global_step} iterations)': (epoch_loss/(self.global_step + 1))})
            pbar.update(self.batch_size)
            self.global_step += self.batch_size
            
            if self.global_step % self.freq == 0:
                print()
                self.logger.log_model_state(self.net, self.global_step)
    
        return epoch_loss 
    
    def process_object(self, true_mask, scan):
        masked_pred = self.net(scan)
        scan = scan.detach().cpu()
        loss = self.criterion(true_mask, masked_pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss 
        
    
    def run_training_loop(self, n_iter, epochs, closure):
        import pdb; pdb.set_trace()
        losses = []
        if self.use_volumes: 
            bs = self.batch_size
        else: 
            bs = 1 
        self.train_loader = DataLoader(self.train, batch_size=bs, shuffle=True, num_workers=8, pin_memory=False)
        self.val_loader = DataLoader(self.val, batch_size=bs, shuffle=False, num_workers=8, pin_memory=False)
        for epoch in range(epochs):
            with tqdm(total=(len(self.train)*197), desc=f'Epoch {epoch + 1}/{epochs}', unit='slices') as pbar:
                epoch_loss = 0
                self.net.train() # tells the net that it is training.
                for volumes in self.train_loader:
                    # (C,197,233,189)
                    mask = volumes['mask'].transpose(1,0)
                    scan = volumes['scan'].transpose(1,0) # (197, 1, 233, 189)
                    epoch_loss = 0 
                    if not self.use_volumes: 
                        epoch_loss += self.process_slices(mask, scan, pbar)
                            
                
                losses.append(epoch_loss)
        return losses
