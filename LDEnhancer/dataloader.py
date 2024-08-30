import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)
def is_img_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".PNG", ".JPG"])

def populate_train_list(images_path):
	ori_dir = 'NAT'
	light_dir = 'NAT_G'
	res_dir = 'NAT_J'
	ori_files = sorted(os.listdir(os.path.join(images_path, ori_dir)))
	light_files = sorted(os.listdir(os.path.join(images_path, light_dir)))
	res_files = sorted(os.listdir(os.path.join(images_path, res_dir)))
	ori_files_path = [os.path.join(images_path, ori_dir, x) for x in ori_files if is_img_file(x)]
	light_files_path = [os.path.join(images_path, light_dir, x) for x in light_files if is_img_file(x)]
	res_files_path = [os.path.join(images_path, res_dir, x) for x in res_files if is_img_file(x)]

	return ori_files_path, light_files_path, res_files_path

class lowlight_loader(data.Dataset):

	def __init__(self, lowlight_images_path, patch_size=128):

		self.ori_files, self.light_files, self.res_files = populate_train_list(lowlight_images_path) 
		self.size = patch_size
		self.img_num = len(self.ori_files)

		print("Total training examples:", self.img_num)


	def __getitem__(self, index):
		tar_index = index % self.img_num
		# ori = torch.from_numpy(np.float32(load_img(self.ori_files[tar_index])))
		ori = Image.open(self.ori_files[tar_index])
		light = Image.open(self.light_files[tar_index])
		res = Image.open(self.res_files[tar_index])
		ori = torch.from_numpy(np.asarray(ori)/255.0).float()
		light = torch.from_numpy(np.asarray(light)/255.0).float()
		res = torch.from_numpy(np.asarray(res)/255.0).float()

		# light = torch.from_numpy(np.float32(load_img(self.light_files[tar_index])))
		# print("###3333")
		ori = ori.permute(2,0,1)
		light = light.permute(2,0,1)
		res = res.permute(2,0,1)
		#Crop Input and Target
		ps = self.size
		H = ori.shape[1]
		W = ori.shape[2]
		# r = np.random.randint(0, H - ps) if not H-ps else 0
		# c = np.random.randint(0, W - ps) if not H-ps else 0
		if H-ps==0:
			r=0
			c=0
		else:
			r = np.random.randint(0, H - ps)
			c = np.random.randint(0, W - ps)
		ori = ori[:, r:r + ps, c:c + ps]
		light = light[:, r:r + ps, c:c + ps]
		res = res[:, r:r + ps, c:c + ps]
		return ori, light, res

	def __len__(self):
		return self.img_num

