'''
    @Date: 01/31/2020
    @Author: Gabriel Nahum 
    @AboutFile: In this file, we will have implementations for different types of U-Nets. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dimension, kernel_size=3):
        super(ConvBlock, self).__init__()
        conv = eval('nn.Conv{}d'.format(dimension)) 
        norm = eval('nn.BatchNorm{}d'.format(dimension))
        
        self.block = nn.Sequential(
                        conv(in_ch, out_ch, kernel_size=kernel_size, padding=int(padding)), 
                        norm(out_ch), 
                        nn.ReLU(inplace=True), 
                        conv(out_ch, out_ch, kernel_size=kernel_size, padding=int(padding)), 
                        norm(out_ch), 
                        nn.ReLU(inplace=True), 
                    )

    def forward(self, x):
        out = self.block(x)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dimension=2, conv_kernel_size=3):
        super(UpBlock, self).__init__()
        
        self.up = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2), nn.Conv2d(in_ch, out_ch, 1))
        

        self.conv_block = ConvBlock(in_ch, out_ch, padding, dimension, kernel_size=conv_kernel_size)
        
        
    def forward(self, x1, x2):
        x1 = self.up(x1) # C x H x W 
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
       
        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)
    
class DownBlock(nn.Module): 
    ''' 
    A single module for the encoding path of a UNet. It will be a max pool followed by a convblock. 
    Args: 
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        padding (bool or int or tupe): adding zero padding on all sized. 
        conv_kernel_size (int or tuple): the size of the window
        pool_kernel_size (int or tuple): the size of the window for the kernel. 
    '''
    def __init__(self, in_ch, out_ch, padding, dimension, conv_kernel_size=3, pool_kernel_size=2): 
        super().__init__()
        pool = eval('nn.MaxPool{}d'.format(dimension))
        
        self.down_block = nn.Sequential(pool(pool_kernel_size),
                                        ConvBlock(in_ch, out_ch, padding, dimension, kernel_size=conv_kernel_size)
                                       )
    def forward(self, x): 
        return self.down_block(x)
    
    
class BasicUNet(nn.Module): 
    '''
        This a U-Net implementation: https://arxiv.org/abs/1505.04597
           
        This class is called BasicUNet because it follows the paper 
        and only requires only the number of input channels and output channels. 
    '''
    def __init__(self, in_channels=1, out_channels=2, wf=64, depth=5, padding=True): 
        
        super(BasicUNet, self).__init__()
        self.dimension = 2 
        self.padding = padding
        self.depth = depth 
        self.down_path = nn.ModuleList()
        self.inc = ConvBlock(in_channels, wf, int(padding), 2, kernel_size=3)
        prev_channels = wf

        for i in range(1, depth):
            fil = wf * (2**i)
            self.down_path.append(
                DownBlock(prev_channels, fil, int(padding), self.dimension, conv_kernel_size=3, pool_kernel_size=2)
            )
            prev_channels = fil

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth-1)):
            fil = wf * (2**i)
            self.up_path.append(
                UpBlock(prev_channels, fil , int(padding), self.dimension, conv_kernel_size=3)
            )
            prev_channels = fil 

        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    
    def forward(self, x): 
#         import pdb; pdb.set_trace()
        blocks = [] 
        x = self.inc(x)
        blocks.append(x)
        for i, down in enumerate(self.down_path): 
            x = down(x)
            if i != len(self.down_path) - 1: 
                blocks.append(x)
        
        for i, up in enumerate(self.up_path): 
            x = up(x, blocks[-i-1])
        
        return self.last(x)
 
    
class UpBlock3D(nn.Module): 
    ''' 
    A single module for the decoding path of a UNet. It will have an upsampling via interpolation followed by a convblock. 
    Args: 
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        padding (bool or int or tupe): adding zero padding on all sized. 
        conv_kernel_size (int or tuple): the size of the window
        mode (str): the mode of interpolation.  
    '''
    def __init__(self, in_ch, out_ch, padding=1, conv_kernel_size=3, mode='nearest'): 
        super(UpBlock3D, self).__init__() 
        
        self.upsample = partial(self._interpolate, mode=mode)
        self.basic_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.joining = partial(self._joining, concat=True)
        self.conv_block = ConvBlock(in_ch, out_ch, padding, 3, kernel_size=conv_kernel_size)
    
    
    def forward(self, x1, x2): 
        output_size = x2.size()[2:]
        x1 = self.upsample(x1, output_size) 
        x1 = self.basic_conv(x1) # 1x1x1x1 convolution. (scale effect)
        x = self.joining(x1, x2) 
        return self.conv_block(x)
    
    @staticmethod
    def _interpolate(x,size,mode): 
        return F.interpolate(x, size=size, mode=mode)
        

    @staticmethod 
    def _joining(encoded, x, concat): 
        if concat: 
            return torch.cat((encoded, x), dim=1)
        else: 
            return encoded + x 

class UNet3D(nn.Module): 
    '''
    3DUnet extended from BasicUNet
    '''
    def __init__(self, in_channels=1, out_channels=2, wf=64, depth=5, padding=True):
        super(UNet3D, self).__init__()
        self.testing = False
        self.dimension = 3 
        self.final_activation = nn.Softmax(dim=1)
        self.padding = int(padding)
        self.depth = depth 
        self.down_path = nn.ModuleList()
        self.inc = ConvBlock(in_channels, wf, padding, 3, kernel_size=3)
        prev_channels = wf 
        
        for i in range(1, depth):
            fil = wf * (2**i)
            self.down_path.append(
                DownBlock(prev_channels, fil, int(padding), 3, 3, 2)
            )
            prev_channels = fil

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth-1)):
            fil = wf * (2**i)
            self.up_path.append(
                UpBlock3D(prev_channels, fil , int(padding), conv_kernel_size=3, mode="nearest")
            )
            prev_channels = fil 

        self.last = nn.Conv3d(prev_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x): 
        blocks = [] 
        x = self.inc(x)
        blocks.append(x)
        for i, down in enumerate(self.down_path): 
            x = down(x)
            blocks.insert(0, x)
        
        blocks.pop(0)
        
        for i, up in enumerate(self.up_path): 
            x = up(x, blocks.pop(0))
        
        x = self.last(x)
        
        if self.testing and self.final_activation is not None: 
            x = self.final_activation(x)
        
        return x 
    
class AttentionBlock(nn.Module): 
    """
    Attention Block 
    """
    def __init__(self, F_g, F_l, F_int): 
        super(AttentionBlock, self).__init__()
        self 
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dimension = 2 
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits