import math
import numpy as np
import torch


class CatMeter:
	'''
	Concatenate Meter for torch.Tensor
	'''
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = None

	def update(self, val):
		if self.val is None:
			self.val = val
		else:
			self.val = torch.cat([self.val, val], dim=0)
	def get_val(self):
		return self.val

	def get_val_numpy(self):
		return self.val.data.cpu().numpy()


class AverageMeter(object):
	def __init__(self):
		self.n = 0
		self.sum = 0.0
		self.var = 0.0
		self.val = 0.0
		self.mean = np.nan
		self.std = np.nan

	def update(self, value, n=1):
		self.val = value
		self.sum += value
		self.var += value * value
		self.n += n

		if self.n == 0:
			self.mean, self.std = np.nan, np.nan
		elif self.n == 1:
			self.mean, self.std = self.sum, np.inf
		else:
			self.mean = self.sum / self.n
			# self.std = math.sqrt(
			# 	(self.var - self.n * self.mean * self.mean) / (self.n - 1.0))

	def value(self):
		return self.mean, self.std

	def reset(self):
		self.n = 0
		self.sum = 0.0
		self.var = 0.0
		self.val = 0.0
		self.mean = np.nan
		self.std = np.nan