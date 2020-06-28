import gzip
import os
import shutil
from pathlib import Path

import numpy as np
import urllib3
import gzip
from matplotlib import pyplot as plt
import cv2	
from PIL import Image
import pandas as pd
# DON'T SUBMIT ...

urls = {
		"train_images": ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz","train"],
		"train_labels": ['http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',"train"],
		"test_images":  ['http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',"test"],
		"test_labels":  ['http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',"test"]
		}



def make_dirs(save_path):
	os.makedirs(os.path.join(save_path,"train"),exist_ok = True)
	os.makedirs(os.path.join(save_path,"train","images"),exist_ok = True)
	os.makedirs(os.path.join(save_path,"test"),exist_ok = True)
	os.makedirs(os.path.join(save_path,"test","images"),exist_ok = True)
	


def download_data(save_dir):
	make_dirs(save_dir)
	# 
	http = urllib3.PoolManager()
	for i in urls:
		
		current_url = urls[i][0]
		current_folder = urls[i][1]
	

		print("Downloading from ",current_url)
		
		file_name = current_url.split("/")[5]
		save_path = Path(os.path.join(save_dir,current_folder,file_name))
		

		with http.request('GET', current_url, preload_content=False) as res, open(save_path, 'wb') as out_file:
			shutil.copyfileobj(res, out_file)
		
		# break




def process_images(data_dir,mode):

	image_size = 28
	if(mode =="train"):

		num_images = 50000
	else:
		num_images = 10000

		
# 
	data_dir = Path(os.path.join(data_dir,mode))
	print(data_dir)
	file_list = os.listdir(data_dir)

	print(file_list)
	image_file = Path(os.path.join(data_dir,file_list[6]))
	label_file = Path(os.path.join(data_dir,file_list[0]))
	image_data = gzip.open(image_file,'r')
	labels_data = gzip.open(label_file,'r')

	image_data.read(16)
	labels_data.read(8)

	labels_list = []
	image_list = []
	for i in range(num_images):
		
		current_image = []
		for j in range(28*28):
			current_image.append(ord(image_data.read(1)))
		
		current_image = np.array(current_image).reshape(28,28)
		current_image = current_image.astype("uint8")

		buf = labels_data.read(1)
		current_label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)[0]
		# print(current_label)
		
		image_name = "{}.png".format(str(i).zfill(8))
		labels_list.append(current_label)
		image_list.append(image_name)

		save_path = Path(os.path.join(data_dir,"images",str(image_name)))
		current_image = Image.fromarray(current_image)
		current_image.save(save_path)

		# break


	print("[INFO] SAVING DATA IN DATAFRAME")
	df = pd.DataFrame({"image_name":image_list, "label":labels_list})
	df.to_csv(os.path.join(data_dir,str(mode) + ".csv"), encoding="utf-8", index=False)








def main():
	# download_data("../../../data/")
	process_images("../../../data/","train")
	# process_images("../../../data/","test")





if __name__ == "__main__":
	main()
