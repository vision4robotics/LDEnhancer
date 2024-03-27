import torch
import torch.nn as nn
# import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import enhance_model 
import Myloss
import dataloader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def train(config):
	os.environ['CUDA_VISIBLE_DEVICES']='0'

	LDEnhancer = enhance_model.enhancer().cuda()
	LDEnhancer.apply(weights_init)
	if config.load_pretrain == True:
	    LDEnhancer.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset =dataloader.lowlight_loader(config.lowlight_images_path)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
	for param in LDEnhancer.UNet1.parameters():
		param.requires_grad=False
	for m in LDEnhancer.UNet1.modules():
		if isinstance(m,nn.BatchNorm2d):
			m.eval()
	L_color = Myloss.L_color()
	L_cen = Myloss.L_cen(16,0.6)
	L_ill = Myloss.L_ill()
	L_perc = Myloss.tracker_perception_loss()
	L_noi = Myloss.noise_loss()
	L_gray = Myloss.Gray_loss()

	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, LDEnhancer.parameters()), lr=config.lr, weight_decay=config.weight_decay)
	
	LDEnhancer.train()

	for epoch in range(config.num_epochs):
		for iteration, img_lowlight in enumerate(train_loader):

			img_lowlight = img_lowlight.cuda()

			enhanced_image,E,S,_,_  = LDEnhancer(img_lowlight)

			Loss_ill = 1600*L_ill(E)
			
			loss_col = 50*torch.mean(L_color(enhanced_image))

			loss_cen = 10*torch.mean(L_cen(enhanced_image))
			
			loss_perc1 =2.6*torch.norm(L_perc(enhanced_image,img_lowlight))
			criterion = nn.L1Loss()
			loss_perc2 = 2.6*torch.norm(L_perc(enhanced_image,L_gray(img_lowlight)))
			
			loss_noise = 50*torch.mean(L_noi(S))

			loss_vis =  Loss_ill   +loss_cen +  loss_col + loss_perc 
			loss_perc = loss_perc1 + loss_perc2
			
			loss = loss_vis +  5*loss_noise
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(LDEnhancer.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("Loss at iteration", iteration+1, ":", loss.item())
				
			if ((iteration+1) % config.snapshot_iter) == 0:
				print("Epoch" + str(epoch) + '.pth')
				print("Loss at epoch", str(epoch), ":", loss.item())
				
				
				torch.save(LDEnhancer.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 		


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="/media/v4r/08981B67981B5294/YLL/IROS2023/train_result/I/train/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=32)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= True)
	parser.add_argument('--pretrain_dir', type=str, default= "pre/Epoch15.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)








	
