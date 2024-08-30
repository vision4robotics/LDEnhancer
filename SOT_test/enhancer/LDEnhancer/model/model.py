import torch
import torch.nn as nn
import torch.nn.functional as F
from .UNet import DownSampleNet,UpSampleNet
from .CrossAttention import CrossAttention

class EstimateNet(nn.Module):
	def __init__(self, number_f = 16):
		super(EstimateNet, self).__init__()
		self.relu = nn.ReLU(inplace=True)
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*2,12,3,1,1,bias=True) 
	def forward(self, x):
		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv3(x2))
		x4 = self.relu(self.e_conv4(torch.cat([x2,x3],1)))
		x_r = F.tanh(self.e_conv5(torch.cat([x1,x4],1)))
		return x_r

class enhancer(nn.Module):
	def __init__(self,device):
		super(enhancer, self).__init__()
		self.downsamplenet = DownSampleNet()

		self.upsamplenet_light = UpSampleNet()
		self.upsamplenet_content = UpSampleNet()

		self.CrossAttention = CrossAttention(device)
		self.decompose = nn.Conv2d(8, 8, 1, 1, 0)

		self.light_estimation = EstimateNet()
		self.content_estimation = EstimateNet()

	def forward(self,x):
		# print(x.shape)
		down_x = self.downsamplenet(x)
		light_down_x = self.decompose(down_x)
		content_down_x = down_x - light_down_x

		light_up_x = self.upsamplenet_light(light_down_x)
		content_x = self.CrossAttention(content_down_x, light_down_x)
		content_up_x = self.upsamplenet_content(content_x)

		Ps = self.light_estimation(light_up_x)
		Pe = self.content_estimation(content_up_x)
		s1,s2,s3,s4 = torch.split(Ps, 3, dim=1)
		e1,e2,e3,e4 = torch.split(Pe, 3, dim=1)
		x = x + (e1 - s1)*(torch.pow(x,2)-x) 
		x = x + (e2 - s2)*(torch.pow(x,2)-x) 
		x = x + (e3 - s3)*(torch.pow(x,2)-x) 
		enhance_image = x + (e4 - s4)*(torch.pow(x,2)-x) 
		A = torch.cat([e1-s1,e2-s2,e3-s3,e4-s4],1)
		return enhance_image, light_up_x, A

if __name__ == "__main__":
	input = torch.rand(2,3,256,256)
	net1 = enhancer()
	output = net1(input)
	print(output.shape)
	
		