'''
    @Date: 01/31/2020
    @Author: Gabriel Nahum 
    @AboutFile: In this file, we will have implementations for different types of U-Nets. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, padding):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=int(padding)), 
                        nn.ReLU(), 
                        nn.BatchNorm2d(out_ch), 
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=int(padding)), 
                        nn.ReLU(), 
                        nn.BatchNorm2d(out_ch)
                    )

    def forward(self, x):
        out = self.block(x)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, padding):
        super(UpBlock, self).__init__()
        
        self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
            )

        self.conv_block = ConvBlock(in_ch, out_ch, padding)
        
        
    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

    
class BasicUNet(nn.Module): 
    '''
        This a U-Net implementation: https://arxiv.org/abs/1505.04597
           
        This class is called BasicUNet because it follows the paper 
        and only requires only the number of input channels and output channels. 
    '''
    def __init__(self, in_channels=3, out_channels=1, wf=64, depth=5, padding=True): 
        
        super(BasicUNet, self).__init__()

        self.padding = padding
        self.depth = depth 
        self.down_path = nn.ModuleList()
        prev_channels = in_channels
      
        for i in range(depth):
            fil = wf * (2**i)
            self.down_path.append(
                ConvBlock(prev_channels, fil, int(padding))
            )
            prev_channels = fil

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            fil = wf * (2**i)
            self.up_path.append(
                UpBlock(prev_channels, fil , int(padding))
            )
            prev_channels = fil 

        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    
    def forward(self, x): 
        blocks = [] 
        for i, down in enumerate(self.down_path): 
            x = down(x)
            if i != len(self.down_path) - 1: 
                blocks.append(x)
                x = F.max_pool2d(x,2)
         
        for i, up in enumerate(self.up_path): 
            x = up(x, blocks[-i-1])
        
        return self.last(x)
    
    