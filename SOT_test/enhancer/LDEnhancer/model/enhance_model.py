import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .CrossAttention import CrossAttention
from .UNet_no_pooling import UNet,Net

class GPNet(nn.Module):

	def __init__(self):
		super(GPNet, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		number_f = 32
		self.e_conv1 = nn.Conv2d(32,number_f,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 


	def forward(self, x):

		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv3(x2))
		x4 = self.relu(self.e_conv4(torch.cat([x2,x3],1)))
		x_r = F.tanh(self.e_conv5(torch.cat([x1,x4],1)))
		return x_r

class LPNet(nn.Module):

	def __init__(self):
		super(LPNet, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		number_f = 32
		
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 

		
	def forward(self, x):
		

		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv3(x2))
		x4 = self.relu(self.e_conv4(torch.cat([x2,x3],1)))
		x_r =F.tanh(self.e_conv5(torch.cat([x1,x4],1))) 
		return x_r

class enhancer(nn.Module):
	def __init__(self):
		super(enhancer, self).__init__()
		
		self.UNet1 = UNet(3)
		self.UNet2 = UNet(3)
		self.CrossAttention = CrossAttention()
		self.Net = Net()

		self.GPNet = GPNet()
		self.LPNet = LPNet()

	def forward(self, x):
		x1, x2 = self.UNet1(x)
		_, x3 = self.UNet2(x1)
		x4 = self.CrossAttention(x2,x3)
		x4 = self.Net(x4,x)
		
		Ps = self.LPNet(x1)
		Pe = self.GPNet(x4)
		s1,s2,s3,s4, s5, s6, s7, s8 = torch.split(Ps, 3, dim=1)
		e1,e2,e3,e4, e5, e6, e7, e8= torch.split(Pe, 3, dim=1)
		x = (x-s1) * (e1+1)
		x = (x-s2) * (e2+1)
		x = (x-s3) * (e3+1)
		enhance_image_1 = (x-s4) * (e4+1)
		x = (enhance_image_1-s5) * (e5+1)
		x = (x-s6) * (e6+1)
		x = (x-s7) * (e7+1)
		enhance_image = (x-s8) * (e8+1)
		e = torch.cat([e1,e2,e3,e4,e5,e6,e7,e8],1)
		s = torch.cat([s1,s2,s3,s4,s5,s6,s7,s8],1)
		return enhance_image,e,s,x1,x4
		

if __name__ == "__main__":
	input = torch.rand(2,3,256,2556)
	net1 = enhancer()
	output = net1(input)
	print(output.shape)
	
		