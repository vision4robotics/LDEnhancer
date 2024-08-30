import torch
import torch.nn as nn
import torch.nn.functional as F

def X2conv(in_channel,out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU())

class DownSampleLayer(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DownSampleLayer, self).__init__()
        #self.x2conv=X2conv(in_channel,out_channel)
        self.conv=nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channel),
        nn.ReLU())
 
    def forward(self,x):
        #out_1=self.x2conv(x)
        out=self.conv(x)
        return out

class UpSampleLayer(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(UpSampleLayer, self).__init__()
        # self.x2conv = X2conv(in_channel, out_channel)
        self.upsample=nn.Sequential(nn.ConvTranspose2d(in_channels=in_channel,out_channels=out_channel,kernel_size=2,stride=2),
        nn.BatchNorm2d(out_channel),
        nn.ReLU())
    def forward(self,x):
        #x=self.x2conv(x)
        x=self.upsample(x)

        return x

class DownSampleNet(nn.Module):
    def __init__(self):
        super(DownSampleNet, self).__init__()
        self.d1=DownSampleLayer(3,4) 
        self.d2=DownSampleLayer(4,4)
        self.d3=DownSampleLayer(4,8)
        self.d4=DownSampleLayer(8,8)
    def forward(self,x):
        out1=self.d1(x)
        out2=self.d2(out1)
        out3=self.d3(out2)
        out4=self.d4(out3)
        return out4

class UpSampleNet(nn.Module):
    def __init__(self):
        super(UpSampleNet, self).__init__()
        self.u1=UpSampleLayer(8,8)
        self.u2=UpSampleLayer(8,4)
        self.u3=UpSampleLayer(4,4)
        self.u4=UpSampleLayer(4,3)

    def forward(self,out4):
        out5=self.u1(out4)
        out6=self.u2(out5)
        out7=self.u3(out6)
        out8=self.u4(out7)
        return out8

if __name__ == "__main__":
    img = torch.randn((2, 3, 256, 256)) 
    net = DownSampleNet()
    net_ = UpSampleNet()
    x,output = net(img)
    # output = net_(x)
    print(output.shape)
    print(x.shape)