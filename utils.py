from collections import defaultdict

import string
import torch

def get_tokens(query):
	return str(query).lower().translate(str.maketrans("", "", string.punctuation)).strip().split()

# Compute the IoU metrics <R@n, IoU=m> for the whole batch
def compute_ious(pm, ps, pe, moment_mask, sm, n = [1, 5], m = [0.1, 0.3, 0.5, 0.7]):
	metrics = defaultdict(lambda: 0.0)
	# metrics = {f"R@{n_}, IoU={m_}": 0.0 for n_ in n for m_ in m}

	# FIX - NMS NOT IMPLEMENTED YET

	# pred_score: B x L x L
	pred_score 		= pm * torch.sqrt(ps.unsqueeze(2)) * torch.sqrt(pe.unsqueeze(1))
	# pred_score: B x L x L
	pred_score		= pred_score * moment_mask
	# pred_score: B x L*L
	pred_score		= pred_score.view(pred_score.shape[0], -1)
	# top_indices: B x top_k
	_, top_indices	= pred_score.topk(k = max(n), dim = 1)
	# top_ious: B x top_k
	top_ious		= torch.gather(sm.view(sm.shape[0], -1), 1, top_indices)

	for n_ in n:
		for m_ in m:
			metrics[f"R@{n_}, IoU={m_}"] += torch.sum((top_ious[:, :n_] > m_).sum(dim = 1) > 0).item()

	return metrics
