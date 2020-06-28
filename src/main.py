from config.defaults import get_cfg_defaults
from trainers.trainer import cls_tripletTrainer
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
	trainer = cls_tripletTrainer(opt)
# 
	# trainer.train()
	trainer.test(400,False)
	# trainer.generate_images()
	


if __name__ == "__main__":
	main()



