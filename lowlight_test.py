import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import enhance_model
import numpy as np
from PIL import Image
import glob
import time
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES']='0'
def lowlight(image_path, LDEnhancer):
	
	data_lowlight = Image.open(image_path)

	data_lowlight = (np.asarray(data_lowlight)/255.0)

	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)
	multiples = 16
	input_ = data_lowlight
	h,w = input_.shape[2], input_.shape[3]
	H,W = ((h+multiples)//multiples)*multiples, ((w+multiples)//multiples)*multiples
	padh = H-h if h%multiples!=0 else 0
	padw = W-w if w%multiples!=0 else 0
	input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
	data_lowlight = input_
	
	start = time.time()
	enhanced_image,_,_,_,_ = LDEnhancer(data_lowlight)
	end_time = (time.time() - start)
	enhanced_image = enhanced_image[:,:,:h,:w]
	print(end_time)
	image_path = image_path.replace('test','result')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	torchvision.utils.save_image(enhanced_image, result_path)
if __name__ == '__main__':
# test_images
	with torch.no_grad():
		filePath = './data/test/'
	
		file_list = os.listdir(filePath)
		LDEnhancer = enhance_model.enhancer().cuda()
		LDEnhancer.load_state_dict(torch.load('snapshots/Epoch25.pth'))
		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
			for image in test_list:
				# image = image
				print(image)
				lowlight(image, LDEnhancer)

