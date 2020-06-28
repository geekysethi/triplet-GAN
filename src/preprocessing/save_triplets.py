import itertools

import os
import random
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

np.random_seed = 42


def create_triplets(data_dir,mode,total_triplets):

	total_classes = 10

	df = pd.read_csv(os.path.join(data_dir,"train",str("train")+".csv"))
	# print(df.head())
	
	labels_list = df["label"].values
	image_list = df["image_name"].values

	image_list, labels_list = shuffle(image_list, labels_list, random_state=42)


	indexes = []

	for current_class in range(total_classes):
		label_indexes = np.random.choice(np.where(labels_list==current_class)[0],int(total_triplets/total_classes),replace = True)
		indexes.append(label_indexes)
	
	flat_list = [item for sublist in indexes for item in sublist]


	# print(indexes)

	triplets_list = []


	for current_class in range(total_classes):

		# print(current_class)

		current_class_list = indexes[current_class]
		new_list = list(set(flat_list) - set(indexes[current_class]))
		anchor_pos_list = list(itertools.product(current_class_list,repeat=2))


		# print(len(new_list))
		# print(len(anchor_pos_list))
		for current_element in new_list:

			for i in anchor_pos_list:
				triplets_list.append([image_list[i[0]], image_list[i[1]], image_list[current_element]])
			
				# break
			# break
		# break

	print(len(triplets_list))


	print("[INFO] SAVING DATA IN DATAFRAME")
	df = pd.DataFrame(triplets_list,columns=["anchor_image","positive_image","negative_image"])
	
	df.to_csv(os.path.join(data_dir,mode,str(mode) + "_triplet_"+str(total_triplets)+".csv"), encoding="utf-8", index=False)



	
if __name__ == "__main__":
	
	create_triplets("../../../data/","train",100)
	create_triplets("../../../data/","train",200)
	
	











