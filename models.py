import torch
import torch.nn as nn

class VideoEncoder(nn.Module):

	def __init__(self, T = 64, d = 512, input_video_dim = 1024):
		super(VideoEncoder, self).__init__()

		self.T 	= T
		self.d 	= d
		self.d0 = input_video_dim
		# CHECK - IS THIS CORRECT?
		# PAPER FORM WHICH CHARADES WAS PICKED USES LINEAR LAYER - self.vid_emb_fn
		# PAPER FROM WHICH OTHER TWO DATASETS WAS PICKED USED 1D CONVOLUTION LAYER - self.vis_conv
		self.ve = nn.Linear(self.d0, self.d) # Video Embedding to convert input video feature space to 512
		self.pe = nn.Embedding(self.T, self.d) # Positional Encoding

	def forward(self, video_features, video_mask):
		x = self.ve(video_features) * video_mask.float()

		p = torch.arange(0, video_mask.shape[1]).unsqueeze(0).long()
		p = self.pe(p) * video_mask.float()

		x += p

		return x

class QueryEncoder(nn.Module):

	def __init__(self, max_query_length = 13, lstm_hidden_size = 256):
		super(QueryEncoder, self).__init__()

		self.lstm = nn.LSTM(input_size = 300, hidden_size = lstm_hidden_size, num_layers = 2, bidirectional = True, batch_first = True)

	def forward(self, query_features):
		x, (h, c) = self.lstm(query_features)
		fs = x[:, -1, :]
		fw = x

		return fs, fw

class Backbone(nn.Module):

	def __init__(self, T = 64, d = 512, input_video_dim = 1024, max_query_length = 13, lstm_hidden_size = 256):
		super(Backbone, self).__init__()

		self.videoencoder = VideoEncoder(T, d, input_video_dim)
		self.queryencoder = QueryEncoder(max_query_length, lstm_hidden_size)

	def forward(self, video_features, video_mask, query_features):
		fv 		= self.videoencoder(video_features, video_mask)
		fs, fw 	= self.queryencoder(query_features)

		f = fv * fs.unsqueeze(1) # Hadamard product of video and sentence features

		return f, fs, fw

class ProposalGeneration(nn.Module):

	def __init__(self, T = 64, L = 16, C = 4):
		super(ProposalGeneration, self).__init__()

		self.T = T
		self.L = L
		self.C = C

	def forward(self, f):
		fc, fm, fb = f, f, f

		return fc, fm, fb
