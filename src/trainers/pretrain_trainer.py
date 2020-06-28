import os
import shutil
import sys
import time
from os import path as osp
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from utils.helpers import load_pretrain_dataloaders, load_model, log_sum_exp
from utils.logger import Logger
from utils.meters import AverageMeter
from utils.losses import TripletLoss
from sklearn.neighbors import KNeighborsClassifier


class Pretrain_Trainer:
	def __init__(self, opt):
		self.opt = opt
		
		torch.manual_seed(opt.SYSTEM.SEED)
		os.makedirs(opt.TRAIN.SAVE_DIR, exist_ok=True)
		
		

		self.unlabeled_loader1, self.unlabeled_loader2 = load_pretrain_dataloaders(self.opt)
		
		self.use_gpu = opt.SYSTEM.USE_GPU

		self._initialize_model()
		self._init_optimizer()

		self.model_dir = os.path.join(self.opt.TRAIN.SAVE_DIR, self.opt.TRAIN.MODEL_NAME)
		sys.stdout = Logger(os.path.join(self.model_dir,'log_train.txt'))
		self.unsupervised_loss_weight = 1
		
		if(self.opt.SYSTEM.DEBUG == False):
			self.experiment = wandb.init(project="dl_endsem")
			
			hyper_params = self.opt
			self.experiment.config.update(hyper_params)
			wandb.watch(self.model_g)
			wandb.watch(self.model_d)
			
		print('=========user config==========')
		print(opt)
		print('============end===============')

	def _initialize_model(self):
		
		self.model_g, self.model_d = load_model(self.opt)


		if(self.use_gpu):
			self.model_g = self.model_g.cuda()
			self.model_d = self.model_d.cuda()


	
	def _init_optimizer(self):
		print("Intializing Optimizers....")
		self.optimizer_g = optim.Adam(self.model_g.parameters(),lr=self.opt.TRAIN.BASE_LR,betas=(0.5,0.999))
		self.optimizer_d = optim.Adam(self.model_d.parameters(),lr=self.opt.TRAIN.BASE_LR,betas=(0.5,0.999))


	def train(self):
		print("PRETRAINING STARTED ...")
		
	
		for epoch in range(self.opt.TRAIN.TOTAL_EPOCHS):


			self.unlabeled_iter2 = iter(self.unlabeled_loader2)
			self.model_d.train()
			self.model_g.train()
			self.train_one_epoch(epoch)

			if((epoch+1) % 100==0 and epoch > 0):

				self.save_model(epoch)			
	
			# break
		
		
		
		if(self.opt.SYSTEM.DEBUG == False):
			print("UPLOADING FINAL FILES ...")
			wandb.save(osp.join(self.model_dir,'/*'))



	def train_one_epoch(self, epoch):

		self.discriminator_losses = AverageMeter()
		self.generator_losses = AverageMeter()

		for batch_idx, data in enumerate(self.unlabeled_loader1):
			
			unlabeled1, _ = data
			unlabeled2, _ = self.unlabeled_iter2.next()
			


			self.train_d(unlabeled1)
			loss_g = self.train_g(unlabeled2)
		
			if(epoch>1 and loss_g>1.0):
				
				print("loop")
				loss_g = self.train_g(unlabeled2)
			self.generator_losses.update(loss_g)
		
			self._update_values(epoch)
		
			# break

		# self._calc_gradients(unlabeled1)
		
		print('Epoch: [{}]\t Discriminator Loss {}\tGenerator Loss {}'.format(epoch,  self.discriminator_losses.mean, self.generator_losses.mean))
		print()

			



	def train_d(self, unlabel):
		
		self.optimizer_d.zero_grad()

		if(self.use_gpu):
			unlabel = unlabel.cuda()

		embedding_unlabel = self.model_d(unlabel)


		fake = self.model_g(unlabel.size()[0])
		fake = fake.view(unlabel.size()).detach()
		embedding_fake = self.model_d(fake)
		
		
		log_unlabel = log_sum_exp(embedding_unlabel)
		log_fake = log_sum_exp(embedding_fake)

		loss_unsupervised = 0.5 * (-torch.mean(log_unlabel) + torch.mean(F.softplus(log_unlabel)) + torch.mean(F.softplus(log_fake))) 
		loss = loss_unsupervised

		loss.backward()
		self.optimizer_d.step()
		self.discriminator_losses.update(loss.item())
	



		return 


	def train_g(self,unlabel):

		self.optimizer_g.zero_grad()
		self.optimizer_d.zero_grad()

		
		if(self.use_gpu):
			unlabel = unlabel.cuda()

		fake = self.model_g(unlabel.size()[0])
		fake = fake.view(unlabel.size())


		features_fake, out_fake = self.model_d(fake,feature=True)
		features_real, out_real = self.model_d(unlabel, feature=True)
		

		features_fake = torch.mean(features_fake,dim=0)
		features_real = torch.mean(features_real,dim=0)


		loss_fr = torch.mean((features_fake - features_real)**2)
		
		loss_fr.backward()
		self.optimizer_g.step()

		return loss_fr.item()

	
	def _calc_gradients(self,unlabel):
		with torch.no_grad():

			if(self.use_gpu):
				unlabel = unlabel.cuda()

			embedding_unlabel = self.model_d(unlabel,feature = True)
			fake = self.model_g(unlabel.size()[0])
			fake = fake.view(unlabel.size()).detach()
			embedding_fake = self.model_d(fake,feature = True)
		
			self.discriminator_real = embedding_unlabel[0].detach().cpu().numpy()
			self.discriminator_fake = embedding_fake[0].detach().cpu().numpy()

			self.generator_bias = self.model_g.fc3.bias.detach().cpu().numpy()
			self.discriminator_weight = self.model_d.layers[-1].weight.detach().cpu().numpy()

			if(self.opt.SYSTEM.DEBUG == False):
				wandb.run.summary.update({"real_feature": wandb.Histogram(self.discriminator_real)})
				wandb.run.summary.update({"fake_feature": wandb.Histogram(self.discriminator_fake)})
				wandb.run.summary.update({"fc3_bias": wandb.Histogram(self.generator_bias)})
				wandb.run.summary.update({"D_feature_weight": wandb.Histogram(self.discriminator_weight)})
			return 



	


	def _update_values(self,epoch):

		if(self.opt.SYSTEM.DEBUG == False):
			self.experiment.log({'Discriminator loss': self.discriminator_losses.mean},step = epoch)
			self.experiment.log({'Generator loss': self.generator_losses.mean},step = epoch)
			

	
	
	
	def save_model(self,epoch):
		print("SAVING MODEL ...")

		torch.save(self.model_g.state_dict(), os.path.join(self.model_dir, "generator_"+str(self.opt.TRAIN.TOTAL_FEATURES)+"_"+str(epoch+1)+".pth"))
		torch.save(self.model_d.state_dict(), os.path.join(self.model_dir, "discriminator_"+str(self.opt.TRAIN.TOTAL_FEATURES)+"_"+str(epoch+1)+".pth"))


	def load_classifier(self):

		print("LOADING MODEL")

		self.model_g.load_state_dict(torch.load(os.path.join(self.model_dir, "generator.pth")))
		self.model_d.load_state_dict(torch.load(os.path.join(self.model_dir, "discriminator.pth")))

		if(self.use_gpu):

			self.model_g.cuda()
			self.model_d.cuda()
