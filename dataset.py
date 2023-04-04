from torch.utils.data import Dataset, DataLoader
from utils import get_tokens

import csv
import h5py
import json
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
		nfeats 				= min(nfeats, self.T)
		out 			 	= np.zeros((self.T, cur_feat.shape[1])) # if T is > number of features, we have 0's for the last frames
		out[:nfeats, :]  	= cur_feat
		return out, nfeats, start_index, end_index

	def collate_fn(self, data):
		tensor_items 	= ["video_features", "video_mask", "query_features", "query_mask", "start_pos", "end_pos"]
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
		}

		return instance

class CharadesSTA(AbstractDataset):

	def __init__(self, data_dir = 'data/charades', T = 64, max_query_length = 13, split = 'train'):
		self.dataset_name 		= "charadessta"
		self.data_dir 			= data_dir
		self.T 					= T
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

	def __init__(self, data_dir = 'data/activitynet', T = 128, max_query_length = 20, split = 'train'):
		self.dataset_name 		= "activitynet"
		self.data_dir 			= data_dir
		self.T 					= T
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

if __name__ == "__main__":
	batch_s = 64
	n_work  = 4

	dataset	= CharadesSTA(split = 'train', T = 64, max_query_length = 13)
	loader 	= DataLoader(dataset, batch_size = batch_s, shuffle = True, collate_fn = dataset.collate_fn, num_workers = n_work)
	st_time = time.time()
	count 	= 0
	for batch in loader:
		assert (batch['video_features'].shape[1] == dataset.T) and (batch['query_features'].shape[1] == dataset.max_query_length)
		count += batch['video_features'].shape[0];
	print(f"# of training samples in CharadesSTA: {count}")
	print(f"Total elapsed time ({(time.time() - st_time):.5f}sec)")

	dataset	= ActivityNet(split = 'train', T = 128, max_query_length = 20)
	loader 	= DataLoader(dataset, batch_size = batch_s, shuffle = True, collate_fn = dataset.collate_fn, num_workers = n_work)
	st_time = time.time()
	count 	= 0
	for batch in loader:
		assert (batch['video_features'].shape[1] == dataset.T) and (batch['query_features'].shape[1] == dataset.max_query_length)
		count += batch['video_features'].shape[0];
	print(f"# of training samples in ActivityNet: {count}")
	print(f"Total elapsed time ({(time.time() - st_time):.5f}sec)")
