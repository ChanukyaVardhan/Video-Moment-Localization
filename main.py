from dataset import CharadesSTA, ActivityNet, TACoS
from models import Backbone
from torch.utils.data import DataLoader
from utils import loss_fn

import argparse
import json
import os
import torch
import yaml

def get_parameters():
	parser 	= argparse.ArgumentParser()
	parser.add_argument("--config_path", default = "config/charadessta.yml", help = "Path to config file.")
	args 	= parser.parse_args()

	with open(args.config_path, "r") as f:
		params = yaml.load(f, Loader=yaml.SafeLoader)
	params["experiment"] = os.path.splitext(os.path.basename(args.config_path))[0]

	return params

def get_datasets(params):
	dataset = None
	if params["dataset"] == "charadessta":
		dataset = CharadesSTA
	elif params["dataset"] == "activitynet":
		dataset = ActivityNet
	elif params["dataset"] == "tacos":
		dataset = TACoS
	else:
		raise Exception(f'Dataset {params["dataset"]} is not a valid dataset!')

	# CHCEK - FIX EVAL DATASET FOR CHARADESSTA BY SPLITTING TRAIN SET.
	train_dataset 	= dataset(params["data_dir"], params["T"], params["max_query_length"], split = "train")
	eval_dataset	= dataset(params["data_dir"], params["T"], params["max_query_length"], split = "test" if params["dataset"] == "charadessta" else "val")

	return train_dataset, eval_dataset

def get_dataloaders(params, train_dataset, eval_dataset):
	train_dataloader 	= DataLoader(
		train_dataset,
		batch_size 	= params["batch_size"],
		shuffle 	= True,
		collate_fn 	= train_dataset.collate_fn,
		num_workers = params["num_workers"],
	)

	eval_dataloader		= DataLoader(
		eval_dataset,
		batch_size 	= params["batch_size"],
		shuffle 	= False,
		collate_fn 	= eval_dataset.collate_fn,
		num_workers = params["num_workers"],
	)

	return train_dataloader, eval_dataloader

def get_model(params):
	model = None
	if params["model"] == "SMIN":
		# LOAD APPROPRIATE MODEL
		model = Backbone(params["T"], params["d"], params["input_video_dim"], params["max_query_length"], params["lstm_hidden_size"])
	else:
		raise Exception(f'Model {params["model"]} is not a valid model!')

	return model

def get_optimizer(model, params):
	optimizer = None
	if params["optimizer"] == "Adam":
		optimizer = torch.optim.Adam(
			model.parameters(),
			lr = params["lr"]
		)
	else:
		raise Exception(f'Optimizer {params["optimizer"]} is not supported!')

	return optimizer

def train_epoch(model, optimizer, train_dataloader, device, params):
	model.train()
	train_loss 		= 0.0
	num_samples 	= 0

	for i, batch in enumerate(train_dataloader):
		video_features 	= batch["video_features"].to(device)
		video_mask 		= batch["video_mask"].to(device)
		query_features 	= batch["query_features"].to(device)
		query_mask 		= batch["query_mask"].to(device)
		start_pos 		= batch["start_pos"].to(device)
		end_pos 		= batch["end_pos"].to(device)

		optimizer.zero_grad()

		outputs 		= model(video_features, video_mask, query_features)

		# UPDATE THIS ACCORDINGLY
		loss 			= loss_fn(outputs, start_pos, end_pos)

		train_loss 	   += loss.item() # FIX THIS
		# COMPUTE IOU AS ACCURACY HERE?

		loss.backward()
		optimizer.step()

		num_samples    += start_pos.shape[0]

	train_loss /= num_samples

	return train_loss

def eval_epoch(model, eval_dataloader, device, params):
	model.eval()
	eval_loss 		= 0.0
	num_samples 	= 0

	for i, batch in enumerate(eval_dataloader):
		video_features 	= batch["video_features"].to(device)
		video_mask 		= batch["video_mask"].to(device)
		query_features 	= batch["query_features"].to(device)
		query_mask 		= batch["query_mask"].to(device)
		start_pos 		= batch["start_pos"].to(device)
		end_pos 		= batch["end_pos"].to(device)

		# FIX THIS APPROPRIATELY
		outputs 		= model(video_features, video_mask, query_features)

		# UPDATE THIS ACCORDINGLY
		loss 			= loss_fn(outputs, start_pos, end_pos)

		eval_loss 	   += loss.item() # FIX THIS
		# COMPUTE IOU AS ACCURACY HERE?

		num_samples    += start_pos.shape[0]

	eval_loss /= num_samples

	return eval_loss

def get_save_paths(params):
	prefix 			= f'{params["checkpoint_path"]}/{params["experiment"]}_'
	model_path 		= f'{prefix}model.pt'
	train_stat_path = f'{prefix}stats.json'

	return model_path, train_stat_path

def get_existing_stats(train_stat_path, start_epoch, params):
	train_stats = {
		"epoch": 		[],
		"train_loss": 	[],
		"eval_loss":	[],
		# ADD OTHER METRICS
	}

	if params["resume_training"] and os.path.exists(train_stat_path):
		existing_stats = json.load(open(train_stat_path, "r"))

		for key, val in existing_stats.items():
			if key in train_stats:
				train_stats[key] = val[:start_epoch - 1]

	return train_stats

def train_model(model, train_dataloader, eval_dataloader, device, params):
	model.train()

	start_epoch = 1
	optimizer 	= get_optimizer(model, params)
	model_path, train_stat_path = get_save_paths(params)
	if params["resume_training"] and os.path.exists(model_path):
		model_details 	= torch.load(model_path)
		start_epoch		= model_details["epoch"] + 1 # Start from the epoch after the checkpoint
		model.load_state_dict(model_details["model"])
		optimizer.load_state_dict(model_details["optimizer"])

	train_stats = get_existing_stats(train_stat_path, start_epoch, params)

	for epoch in range(start_epoch, params["num_epochs"] + 1):
		train_loss 	= train_epoch(model, optimizer, train_dataloader, device, params)
		# HAVE EVAL AFTER A FEW EPOCHS RATHER THAN EVERY EPOCH??
		eval_loss 	= eval_epoch(model, eval_dataloader, device, params)

		# PRINT STATS
		
		train_stats["epoch"].append(epoch)
		train_stats["train_loss"].append(train_loss)
		train_stats["eval_loss"].append(eval_loss)

		with open(train_stat_path, "w") as f:
			json.dump(train_stats, f)

		# SAVE MODEL AND OPTIMIZER ON SOME CONDITION, ALSO SAVE THE CONDITION IN THE PATH AS WELL
		torch.save({
			"epoch": 		epoch,
			"model": 		model.state_dict(),
			"optimizer":	optimizer.state_dict()
		}, model_path)

	return model

if __name__ == "__main__":
	params = get_parameters()

	# Set seed
	torch.manual_seed(params["seed"])
	torch.cuda.manual_seed_all(params["seed"])
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	device	= torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_dataset, eval_dataset 		= get_datasets(params)
	train_dataloader, eval_dataloader 	= get_dataloaders(params, train_dataset, eval_dataset)

	model 	= get_model(params)
	model 	= model.to(device)

	train_model(model, train_dataloader, eval_dataloader, device, params)

	# TEST ACCURACY

	