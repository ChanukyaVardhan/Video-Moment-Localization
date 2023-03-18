from torch.utils.data import Dataset

import csv
import h5py
import numpy as np
import string
import torch
import torchtext
import torch.nn as nn

class CharadesSTA(Dataset):

	vocab 					= torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
	vocab.itos.extend(['<unk>'])
	vocab.stoi['<unk>'] 	= vocab.vectors.shape[0]
	vocab.vectors 			= torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim = 0)
	word_embedding 			= nn.Embedding.from_pretrained(vocab.vectors)

	def __init__(self, data_dir = 'data/charades', T = 64, split = 'train'):
		self.data_dir 		= data_dir
		self.T 				= T
		self.split 			= split

		self.feature_path 	= self.data_dir + "/features/i3d_finetuned/{}.npy"
		ann_path 			= self.data_dir + "/annotations/charades_sta_{}.txt".format(self.split)
		aux_ann_path 		= self.data_dir + "/annotations/Charades_v1_{}.csv".format(self.split)

		self.annotations 	= self._load_annotations(ann_path, aux_ann_path)

	def _load_annotations(self, ann_path, aux_ann_path):
		with open(ann_path, 'r') as f:
			anns   			= f.read().strip().split('\n')

		with open(aux_ann_path) as f:
			reader 			= csv.DictReader(f)
			durations   	= {row['id']: float(row['length']) for row in reader}

		annotations		= []
		for ann in anns:
			info, query 	= ann.split("##")
			vid, spos, epos = info.split(" ")
			duration 		= durations[vid]
			spos 			= float(spos)
			epos 			= min(float(epos), duration) # Found 1805 samples with this issue
			tokens 			= str(query).lower().translate(str.maketrans("", "", string.punctuation)).strip().split()
			token_idx 		= torch.tensor([self.vocab.stoi.get(w if w in self.vocab.stoi else '<unk>') for w in tokens], dtype = torch.long)
			query_features  = self.word_embedding(token_idx)

			if spos < epos: # Found 4 samples with this issue
				annotations.append({
						'video_id': vid,
						'times': [spos, epos],
						'duration': duration,
						'query': query,
						'query_features': query_features,
					})

		return annotations

	def __getitem__(self, index):
		annotation 			= self.annotations[index]
		vid 				= annotation['video_id']
		spos, epos 			= annotation['times']
		duration 			= annotation['duration']
		spos_n 				= spos / duration # Normalized start time
		epos_n 				= epos / duration # Normalized end time
		
		query_features 		= annotation['query_features']

		
		video_features_all 	= np.load(self.feature_path.format(vid)).squeeze()
		video_features, nfeats, start_index, end_index = self.get_fixed_length_features(video_features_all, spos_n, epos_n)
		video_mask 			= np.zeros((self.T, 1))
		video_mask[:nfeats] = 1

		instance 		= {
			'times': annotation['times'], 							# Ground truth start and end pos
			'duration': duration, 									# Duration of the video

			'video_features': torch.FloatTensor(video_features), 	# Uniformly sampled features of size T
			'video_mask': torch.ByteTensor(video_mask), 			# Mask for video features, frames that are less than T will have 0
			'start_pos': torch.FloatTensor([spos_n]), 				# Normalized start position
			'end_pos': torch.FloatTensor([epos_n]), 				# Normalized end position
			'start_index': start_index, 							# Start index in sampled features
			'end_index': end_index, 								# End index in sampled features

			'query_features': query_features, 						# Sentence query features
		}

		# QUERY FEATURES ARE NOT FIXED TO A SENTENCE LENGTH?
		# DO WE NEED A QUERY MASK FOR THE EXTENDED FEATURES?
		# FIX THIS COLLATE_FN OR DO THE SAME LIKE VIDEO MASK?

		return instance

	def get_fixed_length_features(self, feat, start_pos, end_pos):
		nfeats 		= feat.shape[0]
		stride 		= 1.0 if nfeats <= self.T else nfeats * 1.0 / self.T
		spos 		= 0 if self.split != "train" else np.random.randint(0, stride - 0.5 + 1)

		frame_idx 	= np.round(np.arange(spos, nfeats - 0.5, stride)).astype(int)
		start_pos 	= float(nfeats - 1.0) * start_pos # CHECK - IS THIS CORRECT? SHOULDN'T THE 1.0 BE SUBTRACTED OUTSIDE THE PRODUCT?
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

	def __len__(self):
		return len(self.annotations)
