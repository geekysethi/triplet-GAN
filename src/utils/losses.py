import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
	"""
	Triplet loss
	Takes embeddings of an anchor sample, a positive sample and a negative sample
	"""

	def __init__(self):
		super(TripletLoss, self).__init__()

	def forward(self, anchor, positive, negative):

		distance_positive = (torch.sum((anchor - positive).pow(2),dim=1)).pow(0.5)
		distance_negative = (torch.sum((anchor - negative).pow(2),dim=1)).pow(0.5)
		

		distance_positive = distance_positive.view(-1,1)
		distance_negative =distance_negative.view(-1,1)


		z = torch.cat((distance_positive,distance_negative),dim=1)

		return distance_positive, distance_negative, z
		# losses = F.relu(distance_positive - distance_negative + self.margin)
		# return losses.mean() if size_average else losses.sum()