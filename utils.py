import string
import torch
import torch.nn as nn

def get_tokens(query):
	return str(query).lower().translate(str.maketrans("", "", string.punctuation)).strip().split()

def loss_fn(predictions, start_pos, end_pos):
	# FIX THIS,
	return nn.SmoothL1Loss()(predictions[0], predictions[0])
