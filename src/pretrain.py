from config.defaults import get_cfg_defaults
from trainers.pretrain_trainer import Pretrain_Trainer
from utils.helpers import prepare_dirs
import argparse

def main():
	opt = get_cfg_defaults()	
	parser = argparse.ArgumentParser(description="Triplet GAN Training")
	parser.add_argument("--config_file", default="", help="path to config file", type=str)
	
	args = parser.parse_args()

	if args.config_file != "":
		opt.merge_from_file(args.config_file)
	
	opt.freeze()
	# print(opt)



	prepare_dirs(opt)
	trainer = Pretrain_Trainer(opt)
# 
	trainer.train()
	


if __name__ == "__main__":
	main()



