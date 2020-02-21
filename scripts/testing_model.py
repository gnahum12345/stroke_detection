from sd.infra.dataset import AtlasDataset
from sd.infra.logger import Logger 
from sd.models.unet import BasicUNet, UNet3D, UNet 
import torch
from torch.utils.data import DataLoader 


ad = AtlasDataset('/datavol/brain_data/atlas')
first_scan = ad[0]['scan']
first_scan = first_scan.reshape(1, *first_scan.shape) # (1, S, W, H)
first_3d_scan = first_scan.reshape(1,*first_scan.shape) # (1,1,S,W,H)
first_2d_scan = first_scan.transpose(1,0)
scan_dl = DataLoader(first_2d_scan, batch_size=1, shuffle=False)
scan_3dl = DataLoader(ad, batch_size=2, shuffle=False, num_workers=8)

for vol in scan_3dl: 
    break 

scan = vol['scan'].reshape(1,*vol['scan'].shape).transpose(1,0)

print('2d slice ready: ')
for scan_slice in scan_dl: 
    break 


u = UNet(n_channels=1, n_classes=2).double()

try: 
    u_res = u(scan_slice)
    torch.save(u.state_dict(), './u.pt')
    del u_res
    del u 
    print('Successfully went through UNet. Saved in u.pt')
except Exception as e: 
    print(e)
    
bu = BasicUNet(in_channels=1, out_channels=2, wf=64).double()

try: 
    bu_res = bu(scan_slice)
    torch.save(bu.state_dict(), './bu.pt')
    del bu_res 
    del bu
    print("Successfully went through BasicUNet. Saved in bu.pt")
except Exception as e: 
    print(e)
    
import pdb; pdb.set_trace()

u3d = UNet3D(in_channels=1, out_channels=2, wf=6, depth=5, padding=1).double()
u3d = u3d.type(torch.cuda.DoubleTensor)
first_3d_scan = torch.tensor(first_3d_scan.data, requires_grad=False).cuda()

try: 
    with torch.no_grad(): 
        u3d_res = u3d(first_3d_scan)
        u3d_res2 = u3d(scan.cuda())
    torch.save(u3d.state_dict(), './u3d.pt')
    del u3d_res
    print("Successfuly went through 3DUNet. Saved in u3d.pt")
except Exception as e: 
    print(e)




