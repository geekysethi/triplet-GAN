import errno
import math
import os
import pdb
from os import path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from datasets.data_loader import triplet_dataset, unlabeled_dataset
from models.model import Discriminator, Generator


def prepare_dirs(opt):

	print('==> Preparing data..')
	try:
		os.makedirs(os.path.join(opt.TRAIN.SAVE_DIR,opt.TRAIN.MODEL_NAME))
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise



def load_dataloaders(opt):
	print('INITIALIZING TRIPLET DATALOADER ...')

	unlabeled_labels_path = osp.join(opt.DATASET.TRAIN_DIR,"total.csv")
	triplet_labels_path = osp.join(opt.DATASET.TRAIN_DIR,"train","train_triplet_"+str(opt.TRAIN.TOTAL_IMAGES)+".csv")
	train_labels_path = osp.join(opt.DATASET.TRAIN_DIR,"train","train.csv")
	


	triplet_set = triplet_dataset(opt.DATASET.TRAIN_IMAGES_DIR, triplet_labels_path)
	triplet_loader = DataLoader(triplet_set, batch_size = opt.TRAIN.BATCH_SIZE, shuffle=True)


	unlabeled_set = unlabeled_dataset(opt.DATASET.TRAIN_IMAGES_DIR, unlabeled_labels_path)
	unlabeled_loader1 = DataLoader(unlabeled_set, batch_size = opt.TRAIN.BATCH_SIZE, shuffle=True)
	unlabeled_loader2 = DataLoader(unlabeled_set, batch_size = opt.TRAIN.BATCH_SIZE, shuffle=True)
	


	train_set = unlabeled_dataset(opt.DATASET.TRAIN_IMAGES_DIR, train_labels_path)
	train_loader = DataLoader(train_set, batch_size = opt.TRAIN.BATCH_SIZE, shuffle = False)
	

	val_set = unlabeled_dataset(opt.DATASET.VAL_IMAGES_DIR, opt.DATASET.VAL_LABELS)
	val_loader = DataLoader(val_set, batch_size = opt.TRAIN.BATCH_SIZE, shuffle=False)

	print("LENGTH OF UNLABELED DATASET:",len(unlabeled_set))
	print("LENGTH OF TRIPLET DATASET:",len(triplet_set))
	print("LENGTH OF TRAIN DATASET:",len(train_set))
	print("LENGTH OF VALIDATION DATASET:",len(val_set))



	return triplet_loader, unlabeled_loader1, unlabeled_loader2, train_loader, val_loader


def load_pretrain_dataloaders(opt):

	print('INITIALIZING PRETRAIN DATALOADER ...')
	
	train_labels_path = osp.join(opt.DATASET.TRAIN_DIR,"train.csv")
	
	train_set = unlabeled_dataset(opt.DATASET.TRAIN_IMAGES_DIR, train_labels_path)
	unlabeled_loader1 = DataLoader(train_set, batch_size = opt.TRAIN.BATCH_SIZE, shuffle=True)
	unlabeled_loader2 = DataLoader(train_set, batch_size = opt.TRAIN.BATCH_SIZE, shuffle=True)

	print("LENGTH OF DATASET:",len(train_set))

	return unlabeled_loader1, unlabeled_loader2




def load_model(opt):

	print('initializing model ...')
	print("LOADING GENERATOR MODEL")
	model_g = Generator(100,gpu = opt.SYSTEM.USE_GPU)
	print("LOADING DISCRIMINATOR MODEL")
	
	model_d = Discriminator(output_dim=opt.TRAIN.TOTAL_FEATURES, gpu = opt.SYSTEM.USE_GPU)


	return model_g, model_d



def log_sum_exp(x, axis = 1):
	m = torch.max(x, dim = 1)[0]
	return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))


def cosine_dist(x, y):
	'''
	compute cosine distance between two matrix x and y
	with size (n1, d) and (n2, d) and type torch.tensor
	return a matrix (n1, n2)
	'''

	x = F.normalize(x, dim=1)
	y = F.normalize(y, dim=1)
	return torch.matmul(x, y.transpose(0,1))


def plot_tsne(data,labels, save_dir,epoch):
	print("PLOTING TSNE ...")
	palette = sns.color_palette("bright", len(np.unique(labels)))
	tsne_output = TSNE().fit_transform(data)
	# print(n)
	plt.figure(figsize=(16,10))

	sns.scatterplot(tsne_output[:,0],tsne_output[:,1],hue=labels,palette=palette,legend=False)

	os.makedirs(os.path.join(save_dir,"output_images"), exist_ok=True)
	save_dir = os.path.join(save_dir,"output_images","tsne_"+str(epoch)+".png")


	plt.savefig(save_dir,dpi=300)	





def hamming_dist(test_features,train_features):
	n_bits = test_features.shape[1]
	Y = n_bits - np.dot(test_features,np.transpose(train_features)) \
		- np.dot(test_features-1,np.transpose(train_features -1))

	return Y


def mean_average_precison(y_test,y_score):

	all_classes = np.arange(10)
	y_test = label_binarize(y_test, classes=all_classes)

	# print(y_test)

	mAP_list = [];


	for i in all_classes:
		print("Current Class",i)
		mAP_list.append(average_precision_score(y_test[:, i], y_score[:, i]))

	# print(mAP_list)

	return np.mean(mAP_list)


def save_images(images,save_dir):

	for i, current_image in enumerate(images):

		current_image = current_image.reshape(28,28)
		plt.imshow(current_image)

		save_path = os.path.join(save_dir,"output_images","generator_"+str(i)+".png")
		plt.savefig(save_path, dpi= 300)

		if(i==19):
			break


def calc_map(train_features,train_labels,val_features,val_labels):
	
	# print(train_features.shape)
	# print(val_features.shape)

	Y = cdist(val_features,train_features)
	print(Y.shape)
	ind = np.argsort(Y,axis=1)
	
	prec = 0.0;
	acc = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
	# calculating statistics
	for k in range(np.shape(val_features)[0]):

		class_values = train_labels[ind[k,:]]
		
		y_true = (val_labels[k] == class_values)
		y_scores = np.arange(y_true.shape[0],0,-1)
		ap = average_precision_score(y_true, y_scores)
		prec = prec + ap
		for n in range(len(acc)):
			a = class_values[0:(n+1)]
			counts = np.bincount(a)
			b = np.where(counts==np.max(counts))[0]
			if val_features[k] in b:
				acc[n] = acc[n] + (1.0/float(len(b)))
	prec = prec/float(np.shape(val_features)[0])
	acc= [x / float(np.shape(val_features)[0]) for x in acc]
	print("Final results: ")
	print("mAP value: %.4f "% prec)

	return prec
	# for k in range(len(acc)):
	# 	print("Accuracy for %d - NN: %.2f %%" % (k+1,100*acc[k]) )
