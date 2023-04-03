import torch
import torch.nn as nn

def loss_fn(predictions, start_pos, end_pos):
	# FIX THIS,
	return nn.SmoothL1Loss()(predictions[0], predictions[0])
