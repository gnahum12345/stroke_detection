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
    

def get_args(): 
    parser = argparse.ArgumentParser(description="hyperparams for training the modle on images and target masks", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # hparams
    parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs', dest='epochs')
    parser.add_argument('--batch-size', '-b', type=int, nargs='?', default=1, help='Batch size', dest='batchsize')
    parser.add_argument('--learning-rate', '-lr',type=float, nargs='?', default=1e-3, help="Learning rate", dest='lr')
    parser.add_argument('--load', '-l', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--in-channels', '-inc', dest='in_channels', default=3, type=int, help='Number of input channels needed, default is 3')
    parser.add_argument('--n-classes', '-nclass', '-out', dest='out_channels', default=1, type=int, help='Number of output classes in the model.')
    parser.add_argument('--number-of-filters', '-wf', dest='wf', default=6, type=int, help='Number of filters in layer i will be wf*(2**i)')
    parser.add_argument('--padding', '-pad', dest='padding', default=True, type=bool, help='Whether to pad the tensor such that the input shape will be the same as the output. By default this is True')
    
    
    # model 
    parser.add_argument('--model', '-m', dest='model', default="UNet", choices=['UNet'], help='models: ' + ' | '.join(['UNet']))
    
   
    # data_location 
    # TODO (gn): make data_path more robust to minor errors. 
    parser.add_argument('--data_path', '-dp', dest='data_path', default=None, help='path to the given data')
    parser.add_argument('--dataset', '--data', dest='dataset', default='atlas', choices=['atlas', 'isles/17', 'isles/18'], help='datasets: '  + ' | '.join(['atlas', 'isles/17', 'isles/18']))
   
    # logisitics for recovery. 
    parser.add_argument('--debug', dest='debug', type=bool, default=True, help='If this is true, then the setup and hparams will print as well i.e. more intermiedate steps will print')
    parser.add_argument('--logdir', '-ld', dest='logdir', type=str, default=None, help='Location to store the logs, default is ./runs/{DATE}-model')
    parser.add_argument('--use-gpu', '-gpu', dest='gpu', default=True, action='store_true', help='To use a gpu or cpu. ')
    parser.add_argument('--which-gpu', '-gpu-id', dest='gpuid', default=0, help='If using gpu, which gpu? ')
    parser.add_argument('--checkpoint-interval', dest='cp_interval', type=int, default=10, help='Checkpoint interval to save model')
    return parser.parse_args()


def get_model(args): 
    # TODO (gn): adding more models in here. 
    model = None
    if args.model == 'UNet': 
        model = BasicUNet(args.in_channels, args.out_channels, args.wf, args.padding)

    return model

def get_dataset(data_path, dataset): 
    # TODO (gn): add more datasets here: 
    df = None 
    printf('Datapath is: %s' % data_path, bcolors.BOLD)
    printf('Dataset is: %s' % dataset, bcolors.BOLD)
    if dataset == 'atlas': 
        df = AtlasDataset(data_path)
    
    if not df: 
        printf("OH OH dataset is not found!!", bcolors.FAIL + bcolors.BOLD, end='\n')
        printf("Please check the data_path: {0}\n and dataset: {1} \tmatch up.")
        
    return df 

printf = None

def set_debug_mode(debug_mode): 
    global printf
    if debug_mode: 
        printf = printc
    else: 
        printf = lambda x, end='\n': None 
    
def setup_device(args): 
    # todo (gn): make this gpu, after testing. 
    device_tmp = torch.device('cpu') 
    if (torch.cuda.is_available() and args.gpu): 
        torch.cuda.set_device(args.gpuid)
        device = torch.cuda.current_device()
    else: 
        device = torch.device('cpu')
    
    
    return device_tmp


def data_trainer(logger, device, model, dataset): 
    pass 

def main(): 
    args = get_args()
    set_debug_mode(args.debug)
    printf(str(args))
    
    printf('{5}model: {0}\nhparams of model:{6}  \n\tin_channels: {1}\n\tout_channels: {2}\n\tnumber_of_filters: {3} \n\tpadding: {4}'.format(args.model + bcolors.ENDC + bcolors.WHITE, args.in_channels, args.out_channels, args.wf, args.padding, bcolors.HEADER + bcolors.UNDERLINE, bcolors.ENDC), bcolors.BOLD)
    
    model = get_model(args)
    
    printf('Getting dataset: ', bcolors.BOLD) 
    printf('Getting data path', bcolors.UNDERLINE)
    if args.data_path: 
        data_path = args.data_path
    else: 
        printf('Data path is empty, relying on dataset: %s' % args.dataset, bcolors.WARNING)
        base_path = '/datavol/brain_data/'
        data_path = os.path.join(base_path, args.dataset)
    
    printf('Getting pytorch dataset: ', bcolors.UNDERLINE) 

    dataset = get_dataset(data_path, args.dataset)
    
    printf('Got Dataset {}'.format(bcolors.WHITE + bcolors.BOLD + args.dataset), bcolors.CYAN)
    
    printf('Setting up device', bcolors.WHITE)
    device = setup_device(args)
    
    printf('Got device: {}'.format(device), bcolors.WHITE)
    
    printf('Setting up Logger', bcolors.WHITE)
    
    if args.logdir: 
        logger = Logger(args.logdir, args.model)
    else: 
        logger = Logger(None, args.model)
    
    printf('Getting input data')
    sample = dataset[0]
    
#     logger.log_model(model, (sample, ))
    
    # train 
    data_trainer(logger, device, model, dataset)
    
    logger.close()
    
    
    
    

if __name__ == '__main__': 
    main()
    
