import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random

random.seed(1143)


def populate_train_list(lowlight_images_path,G_path):




	image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")
	image_list_lowlight = sorted(image_list_lowlight)
	if G_path:
		G_list = glob.glob(G_path + "*.jpg")
		G_list = sorted(G_list)
		train_list = list(zip(image_list_lowlight,G_list))
	else:
		train_list = image_list_lowlight

	random.shuffle(train_list)

	return train_list

	

class lowlight_loader(data.Dataset):

	def __init__(self, lowlight_images_path,G_path = ""):

		self.train_list = populate_train_list(lowlight_images_path,G_path) 
		self.size = 256
		self.G_path = G_path

		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))


		

	def __getitem__(self, index):		
		
		if self.G_path:
			data_lowlight_path = self.data_list[index][0]		
			data_G_path = self.data_list[index][1]
			data_G = Image.open(data_G_path)
		else:
			data_lowlight_path = self.data_list[index]

		data_lowlight = Image.open(data_lowlight_path)
		
		data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
		data_lowlight = (np.asarray(data_lowlight)/255.0) 
		data_lowlight = torch.from_numpy(data_lowlight).float()
		if self.G_path:
			data_G = data_G.resize((self.size,self.size), Image.ANTIALIAS)
			data_G = (np.asarray(data_G)/255.0)
			data_G = torch.from_numpy(data_G).float()
			data = [data_lowlight.permute(2,0,1),data_G.permute(2,0,1)]
		else:
			data = data_lowlight.permute(2,0,1)

		return data

	def __len__(self):
		return len(self.data_list)

if __name__ == "__main__":
	path = '/media/v4r/08981B67981B5294/YLL/IROS2023/train_result/I/train/'
	G_path = '/media/v4r/08981B67981B5294/YLL/IROS2023/train_result/G/train/'

	input = populate_train_list(path,G_path)
	size =256
	data_list = input
	index = 1
	data_lowlight_path = data_list[index][0]
	data_G_path = data_list[index][1]

	print(input[2])
		
	data_lowlight = Image.open(data_lowlight_path)
	data_lowlight = data_lowlight.resize((size,size), Image.ANTIALIAS)
	data_lowlight = (np.asarray(data_lowlight)/255.0) 
	data_lowlight = torch.from_numpy(data_lowlight).float()

	print(data_lowlight)
	