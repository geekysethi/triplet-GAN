import os
import random
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import itertools

np.random_seed = 42


	
if __name__ == "__main__":

		data_dir = "../../../data"
		train_df = pd.read_csv(os.path.join(data_dir,"train",str("train")+".csv"))
		test_df = pd.read_csv(os.path.join(data_dir,"test",str("test")+".csv"))


		total_df = pd.concat([train_df,test_df])

		print(len(total_df))	

		print("[INFO] SAVING DATA IN DATAFRAME")
		total_df.to_csv(os.path.join(data_dir,"total.csv"), encoding="utf-8", index=False)
