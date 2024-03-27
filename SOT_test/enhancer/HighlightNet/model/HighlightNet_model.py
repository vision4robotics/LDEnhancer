import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import Dropout

class TF(Module):

    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1, activation="relu"):
        super(TF, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.relu = nn.ReLU(inplace=True)
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TF, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        b,c,s=src.permute(1,2,0).size()
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))

    return layers

class enhance_net_nopool(nn.Module):

    def __init__(self):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        
        number_f = 4
        self.e_conv1 = nn.Conv2d(1,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,1,3,1,1,bias=True)

        self.model = nn.Sequential(

            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
        )
        self.TF = TF(16, 8)
        self.F = nn.Conv2d(16, 2, 16, padding=0)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.adapt = nn.AdaptiveAvgPool2d((32,32))



        
    def forward(self, x):
        batch,w,h,b = x.shape
        red,green,blue = torch.split(x ,1,dim = 1)
        v = (red + green + blue)/3
        x1 = self.relu(self.e_conv1(v))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

        v_r = torch.sigmoid(self.e_conv7(torch.cat([x1,x6],1)))
        zero = 0.000001*torch.ones_like(v)
        one = 0.999999*torch.ones_like(v)
        v0 = torch.where(v>0.999999,one,v)
        v0 = torch.where(v<0.000001,zero,v0)
        r = v_r

        v32 = F.interpolate(v,size=(32,32),mode='nearest')
        v1 = self.model(v32)
        bb, cc, ww, hh = v1.size()
        v2 = v1.view(bb,cc,-1).permute(2, 0, 1)
        v3 = self.TF(v2)
        v4 = v3.permute(1,2,0)
        v5 = v4.view(bb,cc,ww,hh)
        level = torch.sigmoid(self.F(v5))

        g1 = level[0,0].item()
        b1 = level[0,1].item()

        g = 0.1*g1+0.2
        b = 0.04*b1+0.06

        for i in range(batch):
            if(i == 0):
                r0 = torch.pow (0.1*level[i,0].item()+0.2,torch.unsqueeze(r[i,:,:,:],0))
            else:
                r1 = torch.pow (0.1*level[i,0].item()+0.2,torch.unsqueeze(r[i,:,:,:],0))
                r0 = torch.cat([r0,r1],0)

        ev0 = torch.pow(v0,r0)

        for i in range(batch):
            if(i == 0):
                L = 400*torch.pow((0.04*level[i,1].item()+0.06 - torch.unsqueeze(v[i,:,:,:],0)),3)

            else:
                L0 = 400*torch.pow((0.04*level[i,1].item()+0.06 - torch.unsqueeze(v[i,:,:,:],0)),3)
                L = torch.cat([L,L0],0)

        L = torch.where(L<0.00001,zero,L)

        ev = ev0 - L

        v = v + 0.000001
        red1 = red/v
        green1 = green/v
        blue1 = blue/v
        red0 = red1*ev
        green0 = green1*ev
        blue0 = blue1*ev
        enhance_image = torch.cat([red0,green0,blue0],1)
        zero1 = torch.zeros_like(x)
        vvv = torch.cat([v,v,v],1)
        t = torch.where(vvv>0.04,zero1,enhance_image)
        A = r
        return enhance_image,A,t



