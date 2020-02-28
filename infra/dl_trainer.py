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
               - optimizer (adam/sgd/agd)
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


    def process_slices(self, mask, scan, pbar, isTraining=True): 
        scan_dl = DataLoader(scan, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        mask_dl = DataLoader(mask, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        epoch_loss = 0 
        for scan_slice, true_mask in zip(scan_dl, mask_dl):
            if (self.cuda): 
                scan_slice = scan_slice.cuda()
                true_mask = true_mask.cuda()
            if isTraining: 
                loss = self.process_object(true_mask, scan_slice)
            else: 
                pred_mask = self.net(scan_slice)
                loss = self.criterion(true_mask, pred_mask)
            epoch_loss += loss.item()
            self.logging(loss.item(), epoch_loss, pbar, isTraining)
            
        return epoch_loss 
    
    def logging(self, loss, epoch_loss, pbar, isTraining): 
        if isTraining: 
            self.logger.scalar_summary('loss/train', loss, self.global_step)
            self.logger.scalar_summary('cum_loss/train', loss, self.global_step)
        else: 
            self.logger.scalar_summary('loss/val', loss, self.global_step)
            self.logger.scalar_summary('cum_loss/val', loss, self.global_step)
            
        pbar.set_postfix(**{f'loss(batch: {self.batch_size})': loss, f'Average loss (across {self.global_step} iterations)': (epoch_loss/(self.global_step + 1))})
        
        pbar.update(self.batch_size)
        self.global_step += self.batch_size
        if self.global_step % self.freq == 0: 
            self.logger.log_model_state(self.net, 'tmp_%d' % self.global_step)
            self.logger.log('model', 'logged latest model', self.global_step)
    
    def process_volumes(self, masks, scans, pbar, isTraining=True): 
        #TODO make the volumes: 
        # (B, S, W, H)
        # convert it to (1, B, S, W, H) => (B, 1, S, W, H)
        masks = masks.unsqueeze(1)
        scans = scans.unsqueeze(1)
        if self.cuda: 
            scans = scans.cuda()
            masks = masks.cuda()
        if isTraining: 
            loss = self.process_object(masks, scans)
            epoch_loss = loss.item()
        else: 
            with torch.no_grad(): 
                masked_pred = self.net(scans)
                loss = self.criterion(masks, masked_pred)
                epoch_loss = loss.item()
            
        self.logging(loss.item(), epoch_loss, pbar, isTraining)
            
        return loss 
    
    def process_object(self, true_mask, scan):
        masked_pred = self.net(scan)
        scan = scan.detach().cpu()
        loss = self.criterion(true_mask, masked_pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss 
        
    
    def run_training_loop(self, epochs, closure):
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
                    # (B,S, W,H) where 1 is the channels. 
                    masks = volumes['mask']
                    scans = volumes['scan'] # (B, 189, 233, 197)
                    if not self.use_volumes: 
                        masks = masks.transpose(1,0)
                        scans = scans.transpose(1,0) # (1,189,233,197)
                        epoch_loss += self.process_slices(masks, scans, pbar, True)
                    else: 
                        epoch_loss += self.process_volumes(masks, scans, pbar, True)
                
                losses.append(epoch_loss)

        self.logger.log_model_state(self.net, 'final_model')
        self.logger.log('model', 'logged last model', self.global_step)
        return losses

#     def log_state(self): 
#         '''
#         A function to log the entire state of the DL_Trainer. 
#         '''
#         # todo 
#     def load_state(self, file): 
#         '''
#         A function that uses the given file to get all the parameters. 
#         '''
#         # todo 
    def validate(self): 
        losses = [] 
        for volumes in self.val_loader: 
            with tqdm(total=len(self.val)*197, desc=f'Volumes', unit='slices') as pbar: 
                masks = volumes['mask']
                scans = volumes['scan']
                if not self.use_volumes: 
                    masks = masks.transpose(1,0)
                    scans = scans.transpose(1,0)
                    losses.append(self.process_slice(masks, scans, pbar, False).item())
                else: 
                    losses.append(self.process_volumes(masks, scans, pbar, False).item())
                    
        return losses 
