import torch
from torch.utils.data import DataLoader, random_split

from sd.infra.dataset import AtlasDataset
from sd.models.unet import * 

def split(data, train,valid): 
    return random_split(data, [int(len(data)*train), int(len(data)*valid), len(data) - int(len(data)*train) - int(len(data)*valid)]) 

# create model
bu = BasicUNet(1,2,64).double()
u = UNet(1,2).double()

# create Dataset
ad = AtlasDataset('/datavol/brain_data/atlas')



train,valid, test = split(ad, 0.7,0.2)

train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

for vol in train_loader: 
    scan = vol['scan']
    mask = vol['mask']
    #scan = scan.transpose(1,0,2,3)
    print('first scan: ', scan.shape) ## (1,197,233,189)
    import pdb; pdb.set_trace()
    scan = scan.transpose(1,0)
    mask = mask.transpose(1,0)
    print('transposed: ', scan.shape)
    scan_dl = DataLoader(scan, batch_size=2, shuffle=False, num_workers=8, pin_memory=True)
    mask_dl = DataLoader(mask, batch_size=2, shuffle=False, num_workers=8, pin_memory=True)
    for scan_slice, true_mask in zip(scan_dl, mask_dl): 
        pred_slice_bu = bu(scan_slice)
        pred_slice_u = u(scan_slice)


