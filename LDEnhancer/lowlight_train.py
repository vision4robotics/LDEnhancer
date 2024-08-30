import torch
import torch.nn as nn
import torchvision
import os
import argparse
import time
import numpy as np
from guided_filter_pytorch.guided_filter import GuidedFilter

import Myloss
import dataloader
from model import model

def train(config):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	LDEnhancer = model.enhancer(device).to(device)

	if config.load_pretrain == True:
	    LDEnhancer.load_state_dict(torch.load(config.pretrain_dir))

	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path,256)		
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

	L_color = Myloss.L_color()
	L_spa = Myloss.L_spa()
	L_exp = Myloss.L_exp_k(16,0.6,1.5)
	L_TV = Myloss.L_TV()
	criterion = nn.SmoothL1Loss()
	optimizer = torch.optim.AdamW(LDEnhancer.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	LDEnhancer.train()

	for epoch in range(config.num_epochs):
		for iteration, batch in enumerate(train_loader):
			img_lowlight = batch[0].cuda()
			img_light = batch[1].cuda()
			img_res = batch[2].cuda()

			enhanced_image, light_up_x, A = LDEnhancer(img_lowlight)

			loss_TV = torch.mean(L_TV(A))
			
			loss_col = 5*torch.mean(L_color(enhanced_image))

			loss_ie = 10*torch.mean(L_exp(enhanced_image))

			loss_spa = 10*torch.mean(L_spa(enhanced_image, img_res)) 

			loss_light = torch.mean(criterion(light_up_x,img_light))

			loss = loss_ie + loss_spa + loss_light + loss_col + loss_TV
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(LDEnhancer.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("Loss at iteration", iteration+1, ":", loss.item())
				# print(f"loss_TV:{loss_TV}\n")
				# print(f"loss_spa:{loss_spa}\n")
				# print(f"loss_col:{loss_col}\n")
				# print(f"loss_ie:{loss_ie}\n")
				# print(f"loss_light:{loss_light}\n")

			if ((iteration+1) % config.snapshot_iter) == 0:
				
				torch.save(LDEnhancer.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 		

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="/mnt/sdb/enhancement")
	parser.add_argument('--lr', type=float, default=0.000001)#0.0001
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--train_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=8)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots_8_13/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)

	train(config)
