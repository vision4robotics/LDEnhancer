import torch
import torch.nn as nn
import torch.nn.functional as F

def X2conv(in_channel,out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU())

class DownsampleLayer(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DownsampleLayer, self).__init__()
        self.x2conv=X2conv(in_channel,out_channel)
        self.conv=nn.Sequential(nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channel),
        nn.ReLU())
        #self.pool=nn.MaxPool2d(kernel_size=2,ceil_mode=True)
 
    def forward(self,x):
        out_1=self.x2conv(x)
        out=self.conv(out_1)
        return out

class UpSampleLayer(nn.Module):
    def __init__(self,in_channel,out_channel):
 
        super(UpSampleLayer, self).__init__()
        self.x2conv = X2conv(in_channel, out_channel)
        self.upsample=nn.Sequential(nn.ConvTranspose2d(in_channels=out_channel,out_channels=out_channel,kernel_size=2,stride=2),
        nn.BatchNorm2d(out_channel),
        nn.ReLU())
    def forward(self,x):
        x=self.x2conv(x)
        x=self.upsample(x)

        return x

class UNet(nn.Module):
    def __init__(self,num_classes):
        super(UNet, self).__init__()
        #下采样
        self.d1=DownsampleLayer(3,16) 
        self.d2=DownsampleLayer(16,32)
        self.d3=DownsampleLayer(32,64)
        self.d4=DownsampleLayer(64,128)
 
        #上采样
        self.u1=UpSampleLayer(128,256)
        self.u2=UpSampleLayer(256,128)
        self.u3=UpSampleLayer(128,64)
        self.u4=UpSampleLayer(64,32)
 
        #输出
        self.x2conv=X2conv(32,32)
        self.final_conv=nn.Conv2d(32,num_classes,kernel_size=1) 
        self._initialize_weights()
 
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
 
    def forward(self,x):
        # 下采样层
        out1=self.d1(x)
        out2=self.d2(out1)
        out3=self.d3(out2)
        out4=self.d4(out3)
 
        # 上采样层 
        out5=self.u1(out4)
        out6=self.u2(out5)
        out7=self.u3(out6)
        out8=self.u4(out7)
 
        # 最后的三层卷积
        out=self.x2conv(out8)
        out=self.final_conv(out)
        return out, out4
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.u1=UpSampleLayer(128,256)
        self.u2=UpSampleLayer(256,128)
        self.u3=UpSampleLayer(128,64)
        self.u4=UpSampleLayer(64,32) 
    def forward(self,out4,x):
        out5=self.u1(out4)
        out6=self.u2(out5)
        out7=self.u3(out6)
        out8=self.u4(out7)
        return out8


 
if __name__ == "__main__":
    img = torch.randn((2, 3, 256, 256)) 
    model = UNet(num_classes=3)
    output, out4 = model(img)
    net = Net()
    x = net(out4,img)
    print(output.shape)
    print(x.shape)