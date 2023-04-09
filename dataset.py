from torch.utils.data import Dataset, DataLoader
from utils import get_tokens

import csv
import h5py
import json
import math
import numpy as np
import os
import time
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F

class AbstractDataset(Dataset):

	# CHECK - IS HAVING A UNK AND PAD THE CORRECT WAY? FIX THIS PROPERLY
	vocab 						= torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
	vocab.itos.extend(['<unk>', '<pad>'])
	vocab.stoi['<unk>'] 		= vocab.vectors.shape[0]
	vocab.stoi['<pad>'] 		= vocab.vectors.shape[0] + 1
	vocab.vectors 				= torch.cat([vocab.vectors, torch.zeros(1, vocab.dim), torch.zeros(1, vocab.dim)], dim = 0)
	word_embedding 				= nn.Embedding.from_pretrained(vocab.vectors)

	def __init__(self, data_dir, T, max_query_length, split):
		pass

	def __len__(self):
		return len(self.annotations)

	def get_query_features(self, query):
		tokens 			= get_tokens(query)
		token_idx 		= torch.tensor([self.vocab.stoi.get(w if w in self.vocab.stoi else '<unk>') for w in tokens], dtype = torch.long)
		token_idx 		= F.pad(token_idx, (0, self.max_query_length - token_idx.shape[0]), value = self.vocab.stoi.get('<pad>'))
		query_features  = self.word_embedding(token_idx)

		return token_idx, query_features

	def get_fixed_length_features(self, feat, start_pos, end_pos):
		nfeats 		= feat.shape[0]
		stride 		= 1.0 if nfeats <= self.T else nfeats * 1.0 / self.T
		if self.split != "train":
			spos = 0
		else:
			random_end = -0.5 + stride
			if random_end == np.floor(random_end):
				random_end = random_end - 1.0
			spos = np.random.randint(0, random_end + 1)

		frame_idx 	= np.round(np.arange(spos, nfeats - 0.5, stride)).astype(int)
		start_pos 	= float(nfeats - 1.0) * start_pos
		end_pos 	= float(nfeats - 1.0) * end_pos

		if not (nfeats < self.T and len(frame_idx) == nfeats) and not (nfeats >= self.T and len(frame_idx) == self.T):
			frame_idx = frame_idx[:self.T] # ignore last one
		assert (nfeats < self.T and len(frame_idx) == nfeats) or (nfeats >= self.T and len(frame_idx) == self.T), \
			"{} != {} or {} != {}".format(len(frame_idx), nfeats, len(frame_idx), self.T)

		start_index, end_index = 0, self.T - 1
		for i in range(len(frame_idx) - 1):
			if frame_idx[i] <= end_pos < frame_idx[i + 1]:
				end_index 	= i
			if frame_idx[i] <= start_pos < frame_idx[i + 1]:
				start_index = i

		cur_feat 		 	= feat[frame_idx, :]
		# SHOULD WE TAKE THE MEAN BETWEEN FRAME INDICES
		# CODE LINKED TO CHARADES STA DOESN'T TAKE MEAN - https://github.com/JonghwanMun/LGI4temporalgrounding/blob/master/src/dataset/abstract_dataset.py#L143
		# CODE LINKED TO TACOS AND ACTIVITYNET TAKES MEAN - https://github.com/microsoft/VideoX/blob/4e24d431e9e56b1aa370d881bd5031c6a8441769/2D-TAN/lib/datasets/__init__.py#L39
		nfeats 				= min(nfeats, self.T)
		out 			 	= np.zeros((self.T, cur_feat.shape[1])) # if T is > number of features, we have 0's for the last frames
		out[:nfeats, :]  	= cur_feat
		return out, nfeats, start_index, end_index

	def collate_fn(self, data):
		tensor_items 	= ["video_features", "video_mask", "query_features", "query_mask", "length_mask", "moment_mask", "start_pos", "end_pos", "sm", "ym", "ss", "ys", "se", "ye", "ya"]
		batch 			= {k: [d[k] for d in data] for k in data[0].keys()}

		if len(data) == 1:
			for k,v in batch.items():
				if k in tensor_items:
					batch[k] = torch.cat(batch[k], 0)
				else:
					batch[k] = batch[k][0]
		else:
			for k in tensor_items:
				batch[k] = torch.cat(batch[k], 0)

		return batch

	def _load_video_features(self, vid):
		NotImplementedError(f'load video features not implemented!')

	def get_iou(self, gt_spos, gt_epos, duration):
		s_times = torch.arange(0, self.L).float() * duration / self.L
		e_times = torch.arange(1, self.L + 1).float() * duration / self.L

		preds 	= torch.stack([
					s_times.repeat_interleave(self.L),
					e_times.repeat(self.L)], dim = 1)
		gts 	= torch.tensor([gt_spos, gt_epos]).unsqueeze(0)

		inter 	= torch.max(torch.tensor(0.0), torch.min(preds[:, 1], gts[:, 1]) - torch.max(preds[:, 0], gts[:, 0]))
		union 	= torch.max(torch.tensor(0.0), torch.max(preds[:, 1], gts[:, 1]) - torch.min(preds[:, 0], gts[:, 0]))

		ious 	= inter / union
		ious 	= ious.reshape(self.L, self.L)

		return ious

	def get_boundary_penalties(self, tau_s, tau_e, duration):
		s_times = torch.arange(0, self.L).float() * duration / self.L
		e_times = torch.arange(1, self.L + 1).float() * duration / self.L

		sigma	= (tau_e - tau_s) / 5.0

		s_s 	= torch.exp(-(s_times - tau_s)**2 / (2.0 * sigma**2))
		s_e 	= torch.exp(-(e_times - tau_e)**2 / (2.0 * sigma**2))

		return s_s, s_e

	def get_snippet_label(self, tau_s, tau_e, duration):
		s_times = torch.arange(0, self.L).float() * duration / self.L
		e_times = torch.arange(1, self.L + 1).float() * duration / self.L

		return torch.logical_and(s_times >= tau_s, e_times <= tau_e)

	def __getitem__(self, index):
		annotation 				= self.annotations[index]
		vid 					= annotation['video_id']
		spos, epos 				= annotation['times']
		duration 				= annotation['duration']
		spos_n 					= spos / duration # Normalized start time
		epos_n 					= epos / duration # Normalized end time
		
		query_features 			= annotation['query_features']
		
		video_features_all 		= self._load_video_features(vid)
		video_features, nfeats, start_index, end_index = self.get_fixed_length_features(video_features_all, spos_n, epos_n)
		video_mask 				= np.zeros((self.T, 1))
		video_mask[:nfeats] 	= 1

		length_mask 			= np.zeros(self.L)
		length_mask[:math.ceil(nfeats / (self.T / self.L))] = 1

		moment_mask 			= np.zeros((self.L, self.L))
		moment_mask[np.triu_indices(self.L)]	= np.logical_and.outer(length_mask, length_mask)[np.triu_indices(self.L)]

		ious 					= self.get_iou(spos, epos, duration)
		ious_label 				= ious > 0.5

		s_s, s_e 				= self.get_boundary_penalties(spos, epos, duration)
		s_s_label 				= s_s > 0.5
		s_e_label 				= s_e > 0.5

		y_a 					= self.get_snippet_label(spos, epos, duration)

		instance 				= {
			'video_id':			vid, 																		# Video Id
			'times': 			annotation['times'], 														# Ground truth start and end pos
			'duration': 		duration, 																	# Duration of the video

			'video_features': 	torch.FloatTensor(video_features).unsqueeze(0),								# Uniformly sampled features of size T
			'video_mask': 		torch.ByteTensor(video_mask).unsqueeze(0), 									# Mask for video features, frames that are less than T will have 0
			'start_pos': 		torch.FloatTensor([spos_n]), 												# Normalized start position
			'end_pos': 			torch.FloatTensor([epos_n]), 												# Normalized end position
			'start_index': 		start_index, 																# Start index in sampled features
			'end_index': 		end_index, 																	# End index in sampled features

			'query_features': 	query_features.unsqueeze(0), 												# Sentence query features
			'query_mask': 		(annotation['token_idx'] < self.vocab.stoi['<pad>']).byte().unsqueeze(0),	# Sentence query mask

			'length_mask': 		torch.BoolTensor(length_mask).unsqueeze(0),									# Mask in the L space
			'moment_mask': 		torch.BoolTensor(moment_mask).unsqueeze(0),									# Mask in the LxL space

			'sm':				ious.unsqueeze(0),															# IoU score of each moment with ground truth
			'ym':				ious_label.unsqueeze(0),													# Binary label of sm with threshold 0.5
			'ss':				s_s.unsqueeze(0), 															# Generated by the unnormalized gaussian
			'ys':				s_s_label.unsqueeze(0), 													# Binary label of ss with threshold 0.5
			'se':				s_e.unsqueeze(0), 															# Generated by the unnormalized gaussian
			'ye':				s_e_label.unsqueeze(0), 													# Binary label of se with threshold 0.5
			'ya':				y_a.unsqueeze(0), 															# Binary label for auxilary snippet matching loss
		}

		return instance

class CharadesSTA(AbstractDataset):

	def __init__(self, data_dir = 'data/charades', T = 64, L = 16, max_query_length = 13, split = 'train'):
		self.data_dir 			= data_dir
		self.T 					= T
		self.L 					= L
		self.max_query_length 	= max_query_length
		self.split 				= split

		self.feature_path 		= self.data_dir + "/features/i3d_finetuned/{}.npy"
		ann_path 				= self.data_dir + "/annotations/charades_sta_{}.txt".format(self.split)
		aux_ann_path 			= self.data_dir + "/annotations/Charades_v1_{}.csv".format(self.split)

		self.annotations 		= self._load_annotations(ann_path, aux_ann_path)

	def _load_annotations(self, ann_path, aux_ann_path):
		with open(ann_path, 'r') as f:
			anns   				= f.read().strip().split('\n')

		with open(aux_ann_path) as f:
			reader 				= csv.DictReader(f)
			durations   		= {row['id']: float(row['length']) for row in reader}

		annotations				= []
		for ann in anns:
			info, query 		= ann.split("##")
			vid, spos, epos 	= info.split(" ")
			duration 			= durations[vid]
			spos 				= max(float(spos), 0)
			epos 				= min(float(epos), duration) # Found 1805 samples with this issue

			if spos < epos: # Found 4 samples with this issue
				token_idx, query_features = self.get_query_features(query)

				annotations.append({
						'video_id': 		vid,
						'times': 			[spos, epos],
						'duration': 		duration,
						'query': 			query,
						'token_idx': 		token_idx,
						'query_features': 	query_features,
					})

		return annotations

	def _load_video_features(self, vid):
		return np.load(self.feature_path.format(vid)).squeeze()

class ActivityNet(AbstractDataset):

	def __init__(self, data_dir = 'data/activitynet', T = 128, L = 64, max_query_length = 20, split = 'train'):
		self.data_dir 			= data_dir
		self.T 					= T
		self.L 					= L
		self.max_query_length 	= max_query_length
		self.split 				= split

		self.feature_path 		= self.data_dir + "/sub_activitynet_v1-3.c3d.hdf5"
		ann_path 				= self.data_dir + "/{}.json".format(self.split)

		self.annotations 		= self._load_annotations(ann_path)

	def _load_annotations(self, ann_path):
		anns = json.load(open(ann_path, "r"))

		annotations = []
		for vid, ann in anns.items():
			duration = ann["duration"]
			for (spos, epos), query in zip(ann["timestamps"], ann["sentences"]):
				spos = max(spos, 0)
				epos = min(epos, duration)
				if spos < epos:
					token_idx, query_features = self.get_query_features(query)

					annotations.append({
							'video_id': 		vid,
							'times': 			[spos, epos],
							'duration': 		duration,
							'query': 			query,
							'token_idx': 		token_idx,
							'query_features': 	query_features,
						})

		return annotations

	def _load_video_features(self, vid):
		return h5py.File(self.feature_path, "r")[vid]['c3d_features'][:]

class TACoS(AbstractDataset):

	def __init__(self, data_dir = 'data/tacos', T = 128, L = 32, max_query_length = 14, split = 'train'):
		self.data_dir 			= data_dir
		self.T 					= T
		self.L 					= L
		self.max_query_length 	= max_query_length
		self.split 				= split

		self.feature_path 		= self.data_dir + "/tall_c3d_features.hdf5"
		ann_path 				= self.data_dir + "/{}.json".format(self.split)

		self.annotations 		= self._load_annotations(ann_path)

	def _load_annotations(self, ann_path):
		anns = json.load(open(ann_path, "r"))

		annotations = []
		for vid, ann in anns.items():
			duration = ann["num_frames"] / ann["fps"]
			for (spos, epos), query in zip(ann["timestamps"], ann["sentences"]):
				spos = max(spos / ann["fps"], 0)
				epos = min(epos / ann["fps"], duration)
				if spos < epos:
					token_idx, query_features = self.get_query_features(query)

					annotations.append({
							'video_id': 		vid,
							'times': 			[spos, epos],
							'duration': 		duration,
							'query': 			query,
							'token_idx': 		token_idx,
							'query_features': 	query_features,
						})

		return annotations

	def _load_video_features(self, vid):
		return h5py.File(self.feature_path, "r")[vid][:]

if __name__ == "__main__":
	batch_s = 64
	n_work  = 4

	dataset	= CharadesSTA(split = 'train', T = 64, L = 16, max_query_length = 13)
	loader 	= DataLoader(dataset, batch_size = batch_s, shuffle = True, collate_fn = dataset.collate_fn, num_workers = n_work)
	st_time = time.time()
	count 	= 0
	for batch in loader:
		assert (batch['video_features'].shape[1] == dataset.T) and (batch['query_features'].shape[1] == dataset.max_query_length)
		count += batch['video_features'].shape[0];
	print(f"# of training samples in CharadesSTA: {count}")
	print(f"Total elapsed time ({(time.time() - st_time):.5f}sec)")

	dataset	= ActivityNet(split = 'train', T = 128, L = 64, max_query_length = 20)
	loader 	= DataLoader(dataset, batch_size = batch_s, shuffle = True, collate_fn = dataset.collate_fn, num_workers = n_work)
	st_time = time.time()
	count 	= 0
	for batch in loader:
		assert (batch['video_features'].shape[1] == dataset.T) and (batch['query_features'].shape[1] == dataset.max_query_length)
		count += batch['video_features'].shape[0];
	print(f"# of training samples in ActivityNet: {count}")
	print(f"Total elapsed time ({(time.time() - st_time):.5f}sec)")

	dataset	= TACoS(split = 'train', T = 128, L = 32, max_query_length = 14)
	loader 	= DataLoader(dataset, batch_size = batch_s, shuffle = True, collate_fn = dataset.collate_fn, num_workers = n_work)
	st_time = time.time()
	count 	= 0
	for batch in loader:
		assert (batch['video_features'].shape[1] == dataset.T) and (batch['query_features'].shape[1] == dataset.max_query_length)
		count += batch['video_features'].shape[0];
	print(f"# of training samples in TACoS: {count}")
	print(f"Total elapsed time ({(time.time() - st_time):.5f}sec)")
