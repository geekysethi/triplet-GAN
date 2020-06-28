import os
import shutil
import sys
import time
from itertools import cycle
from os import path as osp
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from utils.helpers import (cosine_dist, load_dataloaders, load_model,
						   log_sum_exp, mean_average_precison, plot_tsne,
						   save_images,calc_map)
from utils.logger import Logger
from utils.losses import TripletLoss
from utils.meters import AverageMeter, CatMeter



class cls_tripletTrainer:
	def __init__(self, opt):
		self.opt = opt		
		torch.manual_seed(opt.SYSTEM.SEED)

		os.makedirs(os.path.join(opt.TRAIN.SAVE_DIR,opt.TRAIN.MODEL_NAME,str(self.opt.TRAIN.TOTAL_FEATURES)+"_"+str(self.opt.TRAIN.TOTAL_IMAGES)), exist_ok=True)


		self.model_dir = os.path.join(opt.TRAIN.SAVE_DIR, opt.TRAIN.MODEL_NAME, str(self.opt.TRAIN.TOTAL_FEATURES)+"_"+str(self.opt.TRAIN.TOTAL_IMAGES))
		
		os.makedirs(os.path.join(self.model_dir,"logs"),exist_ok=True)
		sys.stdout = Logger(os.path.join(self.model_dir,"logs","log_"+str(time.strftime("%Y%m%d-%H%M%S"))+".txt"))


		self.triplet_loader, self.unlabeled_loader1, self.unlabeled_loader2, self.train_loader, self.val_loader = load_dataloaders(self.opt)
		
		self.use_gpu = opt.SYSTEM.USE_GPU

		self._initialize_model()
		self._init_criterion()
		self._init_optimizer()


		self.unsupervised_loss_weight = 1
		sys.stdout = Logger(os.path.join(self.model_dir,'log_train.txt'))

		
		if(self.opt.SYSTEM.DEBUG == False):
			self.experiment = wandb.init(project="dl_endsem_2")
			
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

		self.load_pretrained_weights()


	
	def _init_optimizer(self):
		print("Intializing Optimizers....")
		self.optimizer_g = optim.Adam(self.model_g.parameters(),lr=self.opt.TRAIN.BASE_LR,betas=(0.5,0.999))
		self.optimizer_d = optim.Adam(self.model_d.parameters(),lr=self.opt.TRAIN.BASE_LR,betas=(0.5,0.999))

	def _init_criterion(self):
		self.triplet_criterion = TripletLoss()
		

	def _calc_triplet_loss(self,anchor, positive, negative):

		distance_positive, distance_negative, z = self.triplet_criterion(anchor, positive, negative)
		z = log_sum_exp(z)

		triplet_loss = -torch.mean(distance_negative) + torch.mean(z)
		

		return triplet_loss



	def train(self):
		

		best_accuracy = 0.0
		best_epoch = 0
		best_precision = 0.0

		# accuracy = []
		# mAP = []
		# triplet_loss = []
		# unsupervised_loss = []

		
		for epoch in range(self.opt.TRAIN.TOTAL_EPOCHS):

			print("TRAINING STARTED ...")
			self.triplet_iter = iter(self.triplet_loader)
			self.unlabeled_iter2 = iter(self.unlabeled_loader2)
			self.model_d.train()
			self.model_g.train()
			self.train_one_epoch(epoch)

			del(self.triplet_iter)
			del(self.unlabeled_iter2)

			if self.opt.TRAIN.EVAL_STEP > 0 and (epoch + 1) % self.opt.TRAIN.EVAL_STEP == 0 or (epoch + 1) == self.opt.TRAIN.TOTAL_EPOCHS:				
				accuracy, mean_precision = self.test(epoch)
				self.save_model(epoch+1)

				if(self.opt.SYSTEM.DEBUG == False):
					self.experiment.log({'accuracy': accuracy},step = epoch)
					self.experiment.log({'mAP': mean_precision},step = epoch)
		

				if(accuracy > best_accuracy):
					best_accuracy = accuracy
					best_precision = mean_precision

					best_epoch = epoch + 1

			# break		
		if(self.opt.SYSTEM.DEBUG == False):
			print("UPLOADING FINAL FILES ...")
			wandb.run.summary["best_accuracy"] = best_accuracy
			wandb.run.summary["best_precision"] = best_precision
			
			# wandb.save(osp.join(self.model_dir,'/*'))



 
		print('Best Accuracy {:.1%}, achived at epoch {}'.format(best_accuracy, best_epoch))


	def train_one_epoch(self, epoch):

		self.triplet_losses = AverageMeter()
		self.unsuperveised_losses = AverageMeter()
		self.discriminator_losses = AverageMeter()
		self.generator_losses = AverageMeter()

		for batch_idx, data in enumerate(self.unlabeled_loader1):
			
			unlabeled1, _ = data
			anchor, positive, negative = next(self.triplet_iter)
			unlabeled2, _ = self.unlabeled_iter2.next()
			


			self.train_d(anchor, positive, negative, unlabeled1)
			loss_g = self.train_g(unlabeled2)
		
			if(epoch>1 and loss_g>1):
				print("loop")
				loss_g = self.train_g(unlabeled2)
			self._update_values(epoch)

			del(unlabeled1)
			del(anchor)
			del(positive)
			del(negative)
			del(unlabeled2)
				
			# self._calc_gradients(unlabeled1)
			# break

		print('Epoch: [{}]\t Discriminator Loss {}\tGenerator Loss {}'.format(epoch,  self.discriminator_losses.mean, self.generator_losses.mean))
		print()

			



	def train_d(self,anchor, positive, negative,unlabel):
		
		self.optimizer_d.zero_grad()

		if(self.use_gpu):
			anchor = anchor.cuda()
			positive = positive.cuda()
			negative = negative.cuda()
			unlabel = unlabel.cuda()

		embedding_anchor = self.model_d(anchor)
		embedding_positive = self.model_d(positive)
		embedding_negative = self.model_d(negative)
		embedding_unlabel = self.model_d(unlabel)

		fake = self.model_g(unlabel.size()[0])
		fake = fake.view(unlabel.size()).detach()
		embedding_fake = self.model_d(fake)
		
		triplet_loss = self._calc_triplet_loss(embedding_anchor,embedding_positive,embedding_negative)

		
		log_unlabel = log_sum_exp(embedding_unlabel)
		log_fake = log_sum_exp(embedding_fake)

		loss_unsupervised = 0.5 * (-torch.mean(log_unlabel) + torch.mean(F.softplus(log_unlabel)) + torch.mean(F.softplus(log_fake))) 
		loss = triplet_loss + self.unsupervised_loss_weight*loss_unsupervised

		loss.backward()
		self.optimizer_d.step()


		self.triplet_losses.update(triplet_loss.item())
		self.unsuperveised_losses.update(loss_unsupervised.item())
		self.discriminator_losses.update(loss.item())

		del(embedding_anchor)
		del(embedding_positive)
		del(embedding_negative)
		del(embedding_unlabel)

		# print(loss.item())

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


		self.generator_losses.update(loss_fr.item())
		

		loss_fr.backward()
		self.optimizer_g.step()

		del(unlabel)
		del(fake)
		del(features_fake)
		del(features_real)

		return loss_fr.item()

		
	def _update_values(self,epoch):

		if(self.opt.SYSTEM.DEBUG == False):
			self.experiment.log({'Triplet loss': self.triplet_losses.mean},step = epoch)
			self.experiment.log({'Unsuperveised loss': self.unsuperveised_losses.mean},step = epoch)
			self.experiment.log({'Discriminator loss': self.discriminator_losses.mean},step = epoch)
			self.experiment.log({'Generator loss': self.generator_losses.mean},step = epoch)

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
				wandb.log({"real_feature": wandb.Histogram(self.discriminator_real)})
				wandb.log({"fake_feature": wandb.Histogram(self.discriminator_fake)})
				wandb.log({"fc3_bias": wandb.Histogram(self.generator_bias)})
				wandb.log({"D_feature_weight": wandb.Histogram(self.discriminator_weight)})
			return 





	def test(self, epoch,tsne):
		self.load_model(epoch)
		print("Testing ...")
		self.evaluate(epoch,tsne)


	def generate_images(self):

		self.model_g.eval()
		unlabeled_iter1 = iter(self.unlabeled_loader1)
		unlabel, _ = unlabeled_iter1.next()
		with torch.no_grad():

		
			if(self.use_gpu):
				unlabel = unlabel.cuda()

			fake = self.model_g(unlabel.size()[0])
			fake = fake.detach().cpu().numpy()


			save_images(fake,self.model_dir)



	def evaluate(self,epoch,tsne = False):
		print("Evaluating ...")


		self.model_d.eval()

		classifier = KNeighborsClassifier(n_neighbors=9)
		train_features_meter, train_labels_meter, = CatMeter(), CatMeter()
		val_features_meter, val_labels_meter, = CatMeter(), CatMeter()
		

		print("Calculating training features ...")
		for batch_idx, data in enumerate(self.train_loader):

			image, label = data

			if(self.use_gpu):
				image = image.cuda()
			
			features = self.model_d(image)

			train_features_meter.update(features.data)	
			train_labels_meter.update(label)

			# break

				
		print("Calculating validation features ...")
		for batch_idx, data in enumerate(self.val_loader):

			image, label = data

			if(self.use_gpu):
				image = image.cuda()
			
			features = self.model_d(image)

			val_features_meter.update(features.data)
			val_labels_meter.update(label)



			# break
		
		train_features = train_features_meter.get_val_numpy()
		train_labels = train_labels_meter.get_val_numpy()
		
		val_features = val_features_meter.get_val_numpy()
		val_labels = val_labels_meter.get_val_numpy()

		mean_precision = calc_map(train_features,train_labels,val_features,val_labels)
		
		print("Training Knn model")
		classifier.fit(train_features,train_labels)
		predict_y = classifier.predict(val_features)
		accuracy = accuracy_score(val_labels, predict_y)


		print("Validation Accuracy",accuracy)
		print("Average Precsion",mean_precision)
		
		if(tsne):
			plot_tsne(train_features,train_labels,self.model_dir,epoch)

		del(classifier)
		del(train_features)
		del(train_labels)
		del(val_features)
		del(val_labels)
		
		return accuracy, mean_precision


	
	def save_model(self,epoch):
		print("SAVING MODEL AT", str(epoch)," ...")

		torch.save(self.model_g.state_dict(), os.path.join(self.model_dir, "generator_"+str(epoch)+".pth"))
		torch.save(self.model_d.state_dict(), os.path.join(self.model_dir, "discriminator_"+ str(epoch) +".pth"))


	def load_model(self,epoch):

		print("LOADING MODEL")

		self.model_g.load_state_dict(torch.load(os.path.join(self.model_dir, "generator_"+str(epoch)+".pth")))
		self.model_d.load_state_dict(torch.load(os.path.join(self.model_dir, "discriminator_"+str(epoch)+".pth")))

		if(self.use_gpu):

			self.model_g.cuda()
			self.model_d.cuda()

	def load_pretrained_weights(self):

		print("LOADING PRETRAINED MODEL	OF",str(self.opt.TRAIN.LOAD_EPOCH), "EPOCH")

		self.model_g.load_state_dict(torch.load(os.path.join(self.opt.TRAIN.SAVE_DIR, "pretrained", "generator_"+str(self.opt.TRAIN.TOTAL_FEATURES)+"_"+str(self.opt.TRAIN.LOAD_EPOCH)+".pth")))
		self.model_d.load_state_dict(torch.load(os.path.join(self.opt.TRAIN.SAVE_DIR, "pretrained", "discriminator_"+str(self.opt.TRAIN.TOTAL_FEATURES)+"_"+str(self.opt.TRAIN.LOAD_EPOCH)+".pth")))

		if(self.use_gpu):

			self.model_g.cuda()
			self.model_d.cuda()
