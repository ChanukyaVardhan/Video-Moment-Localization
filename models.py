import torch
import torch.nn as nn

class VideoEncoder(nn.Module):

	def __init__(self, T = 64, d = 512):
		super(VideoEncoder, self).__init__()

		self.T 	= T
		self.d 	= d
		self.ve = nn.Linear(1024, self.d) # Video Embedding to convert 1024 to 512
		self.pe = nn.Embedding(self.T, self.d) # Positional Encoding

	def forward(self, video_features, video_mask):
		x = self.ve(video_features) * video_mask.float()

		p = torch.arange(0, video_mask.shape[1]).unsqueeze(0).long()
		p = self.pe(p) * video_mask.float()

		x += p

		return x

class QueryEncoder(nn.Module):

	def __init__(self, max_length = 13, lstm_hidden_size = 256):
		super(QueryEncoder, self).__init__()

		self.lstm = nn.LSTM(input_size = 300, hidden_size = lstm_hidden_size, num_layers = 2, bidirectional = True, batch_first = True)

	def forward(self, query_features):
		x, (h, c) = self.lstm(query_features)
		fs = x[:, -1, :]
		fw = x

		return fs, fw

class Backbone(nn.Module):

	def __init__(self, T = 64, d = 512, max_length = 13, lstm_hidden_size = 256):
		super(Backbone, self).__init__()

		self.videoencoder = VideoEncoder(T, d)
		self.queryencoder = QueryEncoder(max_length, lstm_hidden_size)

	def forward(self, video_features, video_mask, query_features):
		fv 		= self.videoencoder(video_features, video_mask)
		fs, fw 	= self.queryencoder(query_features)

		f = fv * fs.unsqueeze(1) # Hadamard product of video and sentence features

		return f, fs, fw

class ProposalGeneration(nn.Module):
	
	# How to deal with the case where L does not divide T?
	# Should Wc be placed on gpu? If so, how?
	def compute_content_matrix(T, L, C):
		Wc = torch.zeros((L, L, C, T))
		for i in range(L):
			for j in range(i, L):
				window_size = (j-i)+1
				window_start, num_frames = i*(T//L), window_size*(T//L)
				clip_size = 1
				if num_frames > C:
					clip_size = num_frames//C
				for c in range(min(C, num_frames)):
					clip_start = window_start + c*clip_size
					X[i, j, c, clip_start : clip_start + clip_size] = 1
		return Wc

	def __init__(self, T = 64, L = 16, C = 4):
		super(ProposalGeneration, self).__init__()

		self.T = T
		self.L = L
		self.C = C
		self.Wc = compute_content_matrix(T, L, C)
		self.avg_pool = nn.AvgPool1d(T//L, stride=T//L)

	def forward(self, f):
		# Wc: (L, L, C, T), f: (B, T, D) -> fc: (B, L, L, C, D)
		fc = torch.einsum('lmit, btj -> blmij', self.Wc, f)
		# fc: (B, L, L, C, D) -> fm: (B, L, L, D)
		fm = torch.sum(fc, dim=2)
		# f: (B, T, D) -> fb: (B, D, T)
		fb = torch.permute(f, (0, 2, 1))
		# fb: (B, D, T) -> fb: (B, D, L)
		fb = self.avg_pool(fb)
		# fb: (B, D, L) -> fb: (B, L, D)
		fb = torch.permute(f, (0, 2, 1))
		return fc, fm, fb
