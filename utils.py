import pdb
import string
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

#from lib2to3.pytree import _Results



def get_tokens(query):
	return str(query).lower().translate(str.maketrans("", "", string.punctuation)).strip().split()

def get_curr_iou(curr_pos, pred_rest_indices):		
	preds 	= pred_rest_indices #[G, 2]
	gts 	= curr_pos.unsqueeze(0) #[1*2]

	inter 	= torch.max(torch.tensor(0.0), torch.min(preds[:, 1], gts[:, 1]) - torch.max(preds[:, 0], gts[:, 0])) #[G]
	union 	= torch.max(torch.tensor(0.0), torch.max(preds[:, 1], gts[:, 1]) - torch.min(preds[:, 0], gts[:, 0])) #[G]

	ious 	= inter / union #[G]	
	return ious

def apply_nms(pred_score, pred_indices, threshold, device = 'cpu'):
    # skipping normalizing indices given it wouldn't matter in ordering    
    #pdb.set_trace()
    print('entered nms')
    results = []
    N = pred_score.shape[0]

	#vectorize later
    for b in range(N):
        keep = []        
		# Sort the detections in descending order of their confidence scores.            
        indices = torch.argsort(pred_score[b], dim=0, descending=True)                
        while len(indices) > 0:
            # Select the detection with the highest confidence score and add its index to the list of indices to be kept.            
            i = indices[0]
            keep.append(i)            
			# calculate iou of the top most window with the rest of the indices and remove the ones which have more overlap with the top window than the threshold
            curr_ious = get_curr_iou(pred_indices[b][i], pred_indices[b,indices[1:],:])
            indices = indices[torch.where(curr_ious <= threshold)[0]]        			
        results.append(torch.tensor(keep))

	#mere padding, stacking to avoid inconsistent tensor sizes
    tensors = [F.pad(tensor, (0, 256 - tensor.size(0)), value=0) for tensor in results]		
    results = torch.stack(tensors, dim=0)
    return results	    

# Compute the IoU metrics <R@n, IoU=m> for the whole batch
def compute_ious(pm, ps, pe, moment_mask, sm, n = [1, 5], m = [0.1, 0.3, 0.5, 0.7], device = 'cpu', nms = False, threshold=0.3):
	#pdb.set_trace()
	metrics = defaultdict(lambda: 0.0)	
	pred_score 		= pm * torch.sqrt(ps.unsqueeze(2)) * torch.sqrt(pe.unsqueeze(1)) #B x L x L	
	pred_score		= pred_score * moment_mask # B x L x L

	#get top_k from NMS instead of just top_k here - get the inputs ready for NMS: indices of predicted boxes(2d input), scores and threshold	
	# remmmeber that iou calculation invovled below, so follow the same indexing
	# s_times = torch.arange(0, self.L).float() * duration / self.L
	# e_times = torch.arange(1, self.L + 1).float() * duration / self.L	
	_, idx2, idx3 = np.meshgrid(np.arange(pred_score.shape[0]), np.arange(pred_score.shape[1]), np.arange(1, pred_score.shape[2]+1), indexing='ij')
	pred_indices  = torch.tensor(np.stack([idx2, idx3], axis=-1)).to(device) #B, L, L, 2	
	pred_indices  = pred_indices.view(pred_indices.shape[0], -1, 2) #B x L*L x
	pred_score	  = pred_score.view(pred_score.shape[0], -1) #B x L*L
	
	#loop over each n and threshold and run nms for each
	for n_ in n:
		for m_ in m:						
			if nms:
				top_indices = apply_nms(pred_score, pred_indices, threshold, device).to(device) #get them in N, k format			
			else: #even if nms is true, bsically verify if the top_indices above have valid k input just in case even when nms is on # but was not necessary
				#fallback to previous method of just getting plain topk
				_, top_indices	= pred_score.topk(k = max(n), dim = 1)	
			#this is where we would need to change if boundary loss is also incorporated				
			top_ious		= torch.gather(sm.view(sm.shape[0], -1), 1, top_indices) 
			metrics[f"R@{n_}, IoU={m_}"] += torch.sum((top_ious[:, :n_] > m_).sum(dim = 1) > 0).item()			
	return metrics

def compute_metrics():
	metrics = defaultdict(lambda: 0.0)
