import torch
import torchvision
import torch.nn.functional as F
import os
import glob
import sys
import time
import numpy as np
import dataloader
from PIL import Image
from model import model

def lowlight(image_path, LDEnhancer):
	
	data_lowlight = Image.open(image_path)
	data_lowlight = (np.asarray(data_lowlight)/255.0)
	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	multiples = 8
	input_ = data_lowlight
	h,w = input_.shape[2], input_.shape[3]
	H,W = ((h+multiples)//multiples)*multiples, ((w+multiples)//multiples)*multiples
	padh = H-h if h%multiples!=0 else 0
	padw = W-w if w%multiples!=0 else 0
	input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
	data_lowlight = input_
	
	start = time.time()
	enhanced_image, light, t, A = LDEnhancer(data_lowlight)
	end_time = (time.time() - start)
	enhanced_image = enhanced_image[:,:,:h,:w][0]

	image_path = image_path.replace('test','result')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
		
	torchvision.utils.save_image(enhanced_image, result_path)


if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	with torch.no_grad():
		filePath = './data/test/'
		file_list = os.listdir(filePath)
		LDEnhancer = model.enhancer(device).cuda()
		LDEnhancer.load_state_dict(torch.load('snapshots/Epoch66.pth'))
		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
			print(test_list)
			for image in test_list:
				# image = image
				print(image)
				lowlight(image, LDEnhancer)

