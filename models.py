import math
from locale import PM_STR
from turtle import forward

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


def compute_content_matrix(T, L, C):
	Wc = torch.zeros((L, L, C, T))
	for i in range(L):
		for j in range(i, L):
			window_size = (j-i)+1
			window_start, num_frames = i*(T//L), window_size*(T//L)
			clip_size = max(1, num_frames // C)
			for c in range(min(C, num_frames)):
				clip_start = window_start + c*clip_size
				Wc[i, j, c, clip_start : clip_start + clip_size] = 1/clip_size
	return Wc
class ProposalGeneration(nn.Module):
	
	# How to deal with the case where L does not divide T?
	# Should Wc be placed on gpu? If so, how?
	# Wc is a upper triangular matrix. Should we make it symmetric as we are using softmax.
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
		fm = torch.mean(fc, dim=3)
		# f: (B, T, D) -> fb: (B, D, T)
		fb = torch.permute(f, (0, 2, 1))
		# fb: (B, D, T) -> fb: (B, D, L)
		fb = self.avg_pool(fb)
		# fb: (B, D, L) -> fb: (B, L, D)
		fb = torch.permute(fb, (0, 2, 1))
		return fc, fm, fb

class Attention(nn.Module):
    def __init__(self, D):
        super(Attention, self).__init__()
        self.D, self.attn_weights = D, None
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
    
    def forward(self, query, key, value):
        # query: (B, Lq, D) -> (B, Lq, D), key: (B, Lk, D) -> (B, Lk, D)  
        query, key = self.W_q(query), self.W_k(key)
        # attn_weights: (B, Lq, Lk)
        attn_weights = torch.matmul(query, torch.transpose(key, 2, 1))/math.sqrt(self.D)
        # attn_weights: (B, Lq, Lk)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # attn_weights: (B, Lq, Lk), value: (B, Lk=Lv, D) -> attn_out: (B, Lq, D)
        attn_out = torch.matmul(attn_weights, value)
        self.attn_weights = attn_weights
        return attn_out

class BoundaryUnit(nn.Module):
    def __init__(self, D):
        super(BoundaryUnit, self).__init__()
        self.D = D
        self.attn_layer = Attention(D)
    
    def forward(self, f_b, f_w, f_s, f_m):
        # f_b: (B, L, D), f_w: (B, Nq, D) -> f_baq: (B, L, D)
        f_baq = self.attn_layer(f_b, f_w, f_w)
        # f_b: (B, L, D), f_s: (B, D), f_baq: (B, L, D) -> f_bq: (B, L, D)
        f_bq = f_b*(f_baq + torch.unsqueeze(f_s, 1))
        # f_bq: (B, L, D) -> A_b: (B, L, L) 
        # is matmul same as bmm?
        A_b = torch.bmm(f_bq, torch.permute(f_bq, (0, 2, 1)))/math.sqrt(self.D)
        # A_b: (B, L, L) -> A_b: (B, L, L) 
        A_b = nn.functional.softmax(A_b, dim=-1)
        # A_b: (B, L, L), f_b: (B, L, D) -> f_bb: (B, L, D)
        f_bb = torch.matmul(A_b, f_b)
        # f_m: (B, L, L, D), f_s: (B, D) -> g_m: (B, L, L, D)
        g_m = torch.sigmoid(f_m * torch.unsqueeze(torch.unsqueeze(f_s, 1), 2))
        # A_b: (B, L, L), f_m: (B, L, L, D) -> f_bm: (B, L, D)
        # I am not confident about the below line.
        f_bm = torch.sum(torch.unsqueeze(A_b, 3) * (g_m*f_m), dim=2)
        return f_bb + f_b + f_bm

class ContentAttention(nn.Module):
    def __init__(self, D):
        super(ContentAttention, self).__init__()
        self.D, self.attn_weights = D, None
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
    
    def forward(self, query, key, value):
        # query: (B, L, L, C, D) -> (B, L, L, C, D), key: (B, Nq, D) -> (B, 1, 1, Nq, D)
        query, key = self.W_q(query), self.W_k(key).unsqueeze(1).unsqueeze(2)
        # query: (B, L, L, C, D), key: (B, 1, 1, Nq, D) -> attn_weights: (B, L, L, C, Nq)
        attn_weights = torch.matmul(query, torch.transpose(key, 3, 4))/math.sqrt(self.D)
        # attn_weights: (B, L, L, C, Nq)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # value: (B, Nq, D) -> (B, 1, 1, Nq, D)
        value = value.unsqueeze(1).unsqueeze(2)
        # attn_weights: (B, L, L, C, Nq), value: (B, 1, 1, Nq, D) -> attn_out: (B, L, L, C, D)
        attn_out = torch.matmul(attn_weights, value)
        self.attn_weights = attn_weights
        return attn_out

class ContentUnit(nn.Module):
    def __init__(self, D, dl):
        super(ContentUnit, self).__init__()
        self.D = D
        self.dl = dl
        # IS CONOVLUTION THE CORRECT WAY TO REDUCE CHANNELS?
        self.down_conv_layer_c = nn.Conv3d(D, dl, 1)
        self.down_conv_layer_w = nn.Conv1d(D, dl, 1)
        self.down_conv_layer_s = nn.Conv1d(D, dl, 1)
        self.up_conv_layer_c = nn.Conv3d(dl, D, 1)
        self.attn_layer = ContentAttention(dl)
    
    def forward(self, f_c, f_w, f_s, f_m):
        # f_c: (B, L, L, C, D) -> f_c_hat: (B, L, L, C, dl)
        f_c_hat = self.down_conv_layer_c(f_c.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        # f_w: (B, Nq, D) -> f_w_hat: (B, Nq, dl)
        f_w_hat = self.down_conv_layer_w(f_w.permute(0, 2, 1)).permute(0, 2, 1)
        # f_s: (B, D) -> f_s_hat: (B, dl)
        f_s_hat = self.down_conv_layer_s(f_s.permute(1, 0)).permute(1, 0)
        # f_c_hat: (B, L, L, C, dl), f_w_hat: (B, Nq, dl) -> f_caq: (B, L, L, C, dl)
        f_caq = self.attn_layer(f_c_hat, f_w_hat, f_w_hat)
        # f_c: (B, L, L, C, dl), f_s_hat: (B, dl), f_caq: (B, L, L, C, dl) -> f_cq: (B, L, L, C, dl)
        f_cq = f_c_hat*(f_caq + f_s_hat.unsqueeze(1).unsqueeze(2).unsqueeze(3))
        # f_cq: (B, L, L, C, dl) -> A_c: (B, L, L, C, C)
        A_c = torch.matmul(f_cq, torch.transpose(f_cq, 3, 4))/math.sqrt(self.dl)
        # A_c: (B, L, L, C, C) -> A_c: (B, L, L, C, C) 
        A_c = nn.functional.softmax(A_c, dim=-1)
        # A_c: (B, L, L, C, C), f_c_hat: (B, L, L, C, dl) -> f_cc_hat: (B, L, L, C, dl)
        f_cc_hat = torch.matmul(A_c, f_c_hat)
        # f_cc_hat: (B, L, L, C, dl) -> f_cc: (B, L, L, C, D)
        f_cc = self.up_conv_layer_c(f_cc_hat.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        # f_m: (B, L, L, D), f_s: (B, D) -> g_m: (B, L, L, D)
        g_m = torch.sigmoid(f_m * f_s.unsqueeze(1).unsqueeze(2))
        # g_m: (B, L, L, D), f_m: (B, L, L, D) -> fbar_m: (B, L, L, D)
        fbar_m = g_m*f_m
        return f_cc + f_c + fbar_m.unsqueeze(3)

class MomentUnit(nn.Module):
	def __init__(self, L):
		super(MomentUnit, self).__init__()
		#need it's own convolutional layer and linear layers
		self.L = L
		#REVIEW THESE CONVOLUTIONAL LAYERS LATER
		self.conv_layer_fb = nn.Conv2d(L, L, 1)
		self.conv_layer_fc = nn.Conv2d(L, L, 1)
		
	def forward(self, f_c, f_m, f_b):
		# f_b: (B, L, D) -> f_b_s: (B, L, 1, D)
		f_b_s = f_b.unsqueeze(2)
		# f_b: (B, L, D) -> f_b_e: (B, 1, L, D)
		f_b_e = f_b.unsqueeze(1)
		# f_c: (B, L, L, C, D) -> f_c_mean: (B, L, L, D)
		f_c_mean = torch.mean(f_c,3)
		#below operations don't chnage dimensions of the inputs		
		conv_fb = self.conv_layer_fb(f_b_s*f_b_e)		
		conv_fc = self.conv_layer_fc(f_c_mean)
		return conv_fb + conv_fc + f_m
class SMI(nn.Module):
	def __init__(self, D, dl, L):
		super(SMI, self).__init__()
		self.D = D
		self.dl = dl
		self.L = L
		self.content_unit = ContentUnit(D, dl)
		self.boundary_unit = BoundaryUnit(D)
		self.moment_unit = MomentUnit(L)		
		#assuming that the weights across all CUs are shared and lly for others
	def forward(self, f_c, f_m, f_b, f_w, f_s):
		#first layer
		cu1 = self.content_unit(f_c, f_w, f_s, f_m)
		bu1 = self.boundary_unit(f_b, f_w, f_s, f_m)
		mu1 = self.moment_unit(cu1, f_m, bu1)
		#second layer
		cu2 = self.content_unit.forward(cu1, f_w, f_s, mu1)
		bu2 = self.boundary_unit.forward(bu1, f_w, f_s, mu1)
		mu2 = self.moment_unit(cu2, mu1, bu2)
		#third layer
		cu3 = self.content_unit.forward(cu2, f_w, f_s, mu2)
		bu3 = self.boundary_unit.forward(bu2, f_w, f_s, mu2)
		mu3 = self.moment_unit(cu3, mu2, bu3)
		return mu3, bu3

class Localization(nn.Module):
	def __init__(self, D):
		super(Localization,self).__init__()		
		self.conv_layer_pm = nn.Conv2d(D, 1, 1)
		self.conv_layer_ps = nn.Conv1d(D, 1, 1)
		self.conv_layer_pe = nn.Conv1d(D, 1, 1)
		self.sigmoid = nn.Sigmoid()
	def forward(self, f_m, f_b):
		# f_m: (B, L, L, D) -> p_m: (B, L, L)
		p_m = self.sigmoid(self.conv_layer_pm(f_m.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)).squeeze(3)
		# f_b: (B, L, D) -> p_s: (B, L)
		p_s = self.sigmoid(self.conv_layer_ps(f_b.permute(0, 2, 1)).permute(0, 2, 1)).squeeze(2)
		# f_b: (B, L, D) -> p_e: (B, L)
		p_e = self.sigmoid(self.conv_layer_pe(f_b.permute(0, 2, 1)).permute(0, 2, 1)).squeeze(2)
		return p_m, p_s, p_e




