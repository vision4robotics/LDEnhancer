import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import dataloader
import enhance_model 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def train(config):
	os.environ['CUDA_VISIBLE_DEVICES']='0,1'

	LDEnhancer = enhance_model.enhancer().cuda()
	LDEnhancer.apply(weights_init)
	if config.load_pretrain == True:
	    LDEnhancer.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path,config.G_path)
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)


	optimizer = torch.optim.Adam(LDEnhancer.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	LDEnhancer.train()

	for epoch in range(config.num_epochs):
		for iteration, data in enumerate(train_loader):

			img_lowlight = data[0]
			G = data[1]

			img_lowlight = img_lowlight.cuda()
			G = G.cuda()

			_,_,_,x1,_  = LDEnhancer(img_lowlight)


			criterion = nn.L1Loss()
	
			
			loss = torch.norm(criterion(G,x1))

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
	parser.add_argument('--G_path', type=str, default="/media/v4r/08981B67981B5294/YLL/IROS2023/train_result/G/train/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=15)
	parser.add_argument('--train_batch_size', type=int, default=28)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="pre_snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "pre_model/Epoch99.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)








	
