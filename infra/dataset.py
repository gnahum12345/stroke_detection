from __future__ import print_function, division 
import os 
import torch
import pandas as pd
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import nibabel as nib
# Ignore warnings 
import warnings 
warnings.filterwarnings('ignore')


class AtlasDataset(Dataset): 
    """Atlas dataset"""
    
    def preprocess(self, i): 
        a = str(i).strip()
        if type(i) == int: 
            a = a.zfill(6)
        return a
    
            
    
    def __init__(self, root_dir, transform=None): 
        '''
        Args: 
            root_dir (string): Directory with all the images + masks. 
            transform (callable, optional): Optional piece of code to be applied on a sample 
        '''
        self.root_dir = root_dir
        
        for f in os.listdir(root_dir): 
            if '.csv' in f: 
                self.mri_df = pd.read_csv(os.path.join(root_dir, f))
                break;        
        path_keys = ['INDI Site ID', 'INDI Subject ID', 'Session'] 
        self.mri_df['path'] = self.mri_df[path_keys].apply(lambda x: '/'.join([self.preprocess(i) for i in x]), axis=1)
        
        self.transform = transform
        
    def __len__(self): 
        return len(self.mri_df)
    
    def __getitem__(self, idx): 
        if torch.is_tensor(idx): 
            idx = idx.tolist()
        
        dir_name = os.path.join(self.root_dir, self.mri_df.iloc[idx]['path'])
        files = os.listdir(dir_name)
    
        # TODO (gn): load the masks more optimally. 
        t1mr, masks = None, None
        for f in files: 
            file_path = os.path.join(dir_name, f); 
            if 't1w' in f and '.nii' in f: 
                t1mr = nib.load(file_path).get_fdata().T
            else: 
                if masks is not None: 
                    masks = masks +  nib.load(file_path).get_fdata().T
                else:                     
                    masks = nib.load(file_path).get_fdata().T
        
        np.true_divide(t1mr, [255.0], out=t1mr)
        np.true_divide(masks, [255.0], out=masks)
        
        sample = {'scan': torch.from_numpy(t1mr), 'mask': torch.from_numpy(masks)}
        if self.transform: 
            sample = self.transform(sample)
        
        return sample 
        
        
