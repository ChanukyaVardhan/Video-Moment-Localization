from dataset import CharadesSTA
from models import Backbone
from torch.utils.data import DataLoader
from utils import loss_fn

import argparse
import json
import torch
import yaml

def get_parameters():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_path", default = "config/charadessta.yml", help = "Path to config file.")
	params = parser.parse_args()

	with open(params.config_path, "r") as f:
		params = yaml.load(f, Loader=yaml.SafeLoader)
	
	return params

def get_datasets(params):
	dataset = None
	if params["dataset"] == "charadessta":
		dataset = CharadesSTA
	else:
		raise Exception(f'Dataset {params["dataset"]} is not a valid dataset!')

	train_dataset 	= dataset(params["data_dir"], params["T"], params["max_query_length"], split = "train")
	eval_dataset	= dataset(params["data_dir"], params["T"], params["max_query_length"], split = "test")

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
		model = Backbone(params["T"], params["d"], params["max_query_length"], params["lstm_hidden_size"])
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

		train_loss 	   += loss # FIX THIS
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

		outputs 		= model(video_features, video_mask, query_features)

		# UPDATE THIS ACCORDINGLY
		loss 			= loss_fn(outputs, start_pos, end_pos)

		eval_loss 	   += loss # FIX THIS
		# COMPUTE IOU AS ACCURACY HERE?

		num_samples    += start_pos.shape[0]

	eval_loss /= num_samples

	return eval_loss

def get_model_and_optimizer_path(params):
	# FIX THIS WITH SOME EXPERIMENT IDENTIFER
	return params["checkpoint_path"] + "/model_best.pt", params["checkpoint_path"] + "/optimizer_best.pt"

def train_model(model, train_dataloader, eval_dataloader, device, params):
	model.train()

	optimizer 	= get_optimizer(model, params)
	# LOAD OPTIMIZER FROM EXISTISTING CHECKPOINT IF AVAILABLE

	for epoch in range(1, params["num_epochs"] + 1):
		train_loss 	= train_epoch(model, optimizer, train_dataloader, device, params)
		# HAVE EVAL AFTER A FEW EPOCHS RATHER THAN EVERY EPOCH??
		eval_loss 	= eval_epoch(model, eval_dataloader, device, params)
		# PRINT STATS

		# SAVE MODEL AND OPTIMIZER ON SOME CONDITION
		model_path, optimizer_path = get_model_and_optimizer_path(params)
		torch.save(model.state_dict(), model_path)
		torch.save(optimizer.state_dict(), optimizer_path)

	# LOAD THE BEST MODEL HERE BEFORE RETURNING?
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
	# LOAD MODEL FROM EXISTING CHECKPOINT IF AVAILABLE

	train_model(model, train_dataloader, eval_dataloader, device, params)

	# EVAL MODEL

	