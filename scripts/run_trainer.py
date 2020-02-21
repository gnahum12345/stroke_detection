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
from sd.infra.losses import *
from sd.infra.dl_trainer import *
from sd.models.unet import *
from sd.infra.global_utils import *


def get_args():
    parser = argparse.ArgumentParser(description="hyperparams for training the modle on images and target masks", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # hparams
    parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs', dest='epochs')
    parser.add_argument('--batch-size', '-b', type=int, nargs='?', default=1, help='Batch size', dest='batch_size')
    parser.add_argument('--learning-rate', '-lr',type=float, nargs='?', default=1e-3, help="Learning rate", dest='lr')
    parser.add_argument('--load', '-l', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.2, help='Percent of the data that is used as validation (0-1), default 0.2')
    parser.add_argument('--in-channels', '-inc', dest='in_channels', default=1, type=int, help='Number of input channels needed, default is 1')
    parser.add_argument('--n-classes', '-nclass', '-out', dest='out_channels', default=2, type=int, help='Number of output classes in the model. Default 2 (binary)')
    parser.add_argument('--number-of-filters', '-wf', dest='wf', default=64, type=int, help='Number of filters in layer i will be wf*(2**i), default 64')
    parser.add_argument('--padding', '-pad', dest='padding', default=True, type=bool, help='Whether to pad the tensor such that the input shape will be the same as the output. By default this is True')


    # model
    parser.add_argument('--model', '-m', dest='model', default='UNet', choices=['UNet', 'BasicUNet', '3DUNet'], help='models: ' + ' | '.join(['UNet', 'BasicUNet', '3DUNet']))


    # data_location
    # TODO (gn): make data_path more robust to minor errors.
    parser.add_argument('--data_path', '-dp', dest='data_path', default=None, help='path to the given data')
    parser.add_argument('--dataset', '--data', dest='dataset', default='atlas', choices=['atlas', 'isles/17', 'isles/18'], help='datasets: '  + ' | '.join(['atlas', 'isles/17', 'isles/18']))

    # optimizer_type:
    parser.add_argument('--optimizer', '-op', dest='optimizer', default='adam', choices=['adam', 'sdg'], help='optimizers: ' + ' | '.join(['adam', 'sdg']) +  '\nfor more info see: https://pytorch.org/docs/stable/optim.html')

    # logisitics for recovery.
    parser.add_argument('--seed', '-s', dest='seed', type=int, default=0, help="Random seed to make the program reproducibile, by default 0")
    parser.add_argument('--debug','-d', dest='debug', type=bool, default=True, help='If this is true, then the setup and hparams will print as well i.e. more intermiedate steps will print')
    parser.add_argument('--logdir', '-ld', dest='logdir', type=str, default=None, help='Location to store the logs, default is ./runs/{DATE}-model')
    parser.add_argument('--use-gpu', '-gpu', '-g', dest='gpu', default=False, action='store_true', help='To use a gpu or cpu. ')
    parser.add_argument('--which-gpu', '--gpu-id', '-gid', dest='gpuid', default=0, help='If using gpu, which gpu? ' + ' '.join(str(i) for i in range(torch.cuda.device_count())))
    parser.add_argument('--checkpoint-interval', '-freq', dest='freq', type=int, default=10, help='Checkpoint interval to save model')
    return parser.parse_args()


def get_model(args):
    # TODO (gn): adding more models in here.
    model = None
    if args.model == 'BasicUNet':
        model = BasicUNet(in_channels=args.in_channels, out_channels=args.out_channels, wf=args.wf, padding=args.padding, dimension=2).double()
    elif args.model == 'UNet': 
        model = UNet(args.in_channels, args.out_channels).double()
    elif args.model == '3DUNet': 
        model = BasicUNet(in_channels=args.in_channels, out_channels=args.out_channels, wf=args.wf, padding=args.padding, dimension=3).double()

    assert model is not None, 'Need to specify a model'
    if args.load: 
        try:   
            model.load_state_dict(torch.load(args.load))
        except Exception as e: 
            printf(str(e), bcolors.FAIL)
      
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
        printf("Please check the data_path: {0}\n and dataset: {1} \tmatch up.".format(data_path, dataset))

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
    if torch.cuda.is_available() and args.gpu:
        device = torch.device(int(args.gpuid))
    else:
        device = torch.device('cpu')
    

    return device

    

def get_dl_params(args, logger, device, model, dataset):
    '''
    Gets the params for DL_Trainer.
    [params, logger, device, net,
    dataset, optimizer_type, lr, freq, seed, loss_fn]
    '''
    ns = argparse.Namespace()
    ns.args = args
    ns.logger = logger
    ns.device = device
    ns.model = model
    ns.dataset = dataset
    ns.optimizer_type = args.optimizer
    ns.lr = args.lr
    ns.freq = args.freq
    ns.seed = args.seed
    ns.batch_size = args.batch_size
    ns.loss_fn = dice_loss
    ns.epochs = args.epochs
    return ns


def log_model(logger, dataset, model): 
    ''' 
    preprocess the data to fit the 2D/3D models
    and then logs it. 
    ''' 
#     import pdb; pdb.set_trace()
    sample_scan = dataset[0]['scan'] # (T, W, H)
    if model.dimension == 2: 
        # this is a 2d model. 
        sample_slice = sample_scan[0] # (W,H)
        sample = sample_slice.reshape(1, 1, *sample_slice.shape) # (B, C, W, H)
        logger.log_model(model.cpu(), (sample.cpu(), ))
    elif model.dimension == 3: 
        #this is a 3d model 
        sample = sample_scan.reshape(1, 1,*sample_scan.shape) # (B, C, D, W, H)
        logger.log_model(model.cpu(), (sample.cpu(), ))
        

def main():
    args = get_args()
    set_debug_mode(args.debug)
    printf(str(args))
    
    printf('Setting up device', bcolors.WHITE)
    device = setup_device(args)
    printf('Got device: {}'.format(device), bcolors.WHITE)

    printf('{5}model: {0}\nhparams of model:{6}  \n\tin_channels: {1}\n\tout_channels: {2}\n\tnumber_of_filters: {3} \n\tpadding: {4}'.format(args.model + bcolors.ENDC + bcolors.WHITE, args.in_channels, args.out_channels, args.wf, args.padding, bcolors.HEADER + bcolors.UNDERLINE, bcolors.ENDC), bcolors.BOLD)

    model = get_model(args)
    model.to(device)
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

    printf('Setting up Logger', bcolors.WHITE)

    if args.logdir:
        logger = Logger(args.logdir, args.model)
    else:
        logger = Logger(None, args.model)
 
    printf('Logging to %s' % (logger.writer.get_logdir()), bcolors.WHITE)

    printf('Logging model')
    log_model(logger, dataset, model)
    
    params = get_dl_params(args, logger, device, model, dataset)
#    import pdb; pdb.set_trace()
    dl = DL_Trainer(params)
    dl.run_training_loop(args.epochs, 1, lambda x: x)
    logger.close()





if __name__ == '__main__':
    main()
