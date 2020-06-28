import glob
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils



transform = transforms.Compose([transforms.ToTensor()])
# transform_test = transforms.Compose([transforms.ToTensor()])


class triplet_dataset(Dataset):

	def __init__(self,image_dir, labels_path, transforms=transform):
		
		self.df = pd.read_csv(labels_path)
		self.image_dir = image_dir
		self.transforms = transforms


	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		anc_image = cv2.imread(os.path.join(self.image_dir,self.df["anchor_image"][index]),0)
		pos_image = cv2.imread(os.path.join(self.image_dir,self.df["positive_image"][index]),0)
		neg_image = cv2.imread(os.path.join(self.image_dir,self.df["negative_image"][index]),0)

		# anc_label = self.df["anchor_label"][index]
		# pos_label = self.df["positive_label"][index]
		# neg_label = self.df["negative_label"][index]

		
		anc_image = self.transforms(anc_image)
		pos_image = self.transforms(pos_image)
		neg_image = self.transforms(neg_image)

		anc_image = anc_image.view(784)
		pos_image = pos_image.view(784)
		neg_image = neg_image.view(784)


		return anc_image, pos_image, neg_image






class unlabeled_dataset(Dataset):

	def __init__(self,image_dir, labels_path, transforms=transform):
		
		self.df = pd.read_csv(labels_path)
		self.image_dir = image_dir
		self.transforms = transforms

		


	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		unlabeled_image = cv2.imread(os.path.join(self.image_dir,self.df["image_name"][index]),0)

		label = self.df["label"][index]

		
		
		unlabeled_image = self.transforms(unlabeled_image)
		unlabeled_image = unlabeled_image.view(784)

		return unlabeled_image,label



