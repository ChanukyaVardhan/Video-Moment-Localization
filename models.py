import math

import torch
import torch.nn as nn


class VideoEncoder(nn.Module):

        def __init__(self, T = 64, d = 512, input_video_dim = 1024, device = 'cpu'):
                super(VideoEncoder, self).__init__()

                self.T          = T
                self.d          = d
                self.d0         = input_video_dim
                self.device = device
                # CHECK - IS THIS CORRECT?
                # PAPER FORM WHICH CHARADES WAS PICKED USES LINEAR LAYER - self.vid_emb_fn
                # PAPER FROM WHICH OTHER TWO DATASETS WAS PICKED USED 1D CONVOLUTION LAYER - self.vis_conv
                # CHECK - DO NEED A NON LINEAR ACTIVATION HERE?
                # Video Embedding to convert input video feature space to 512
                self.ve         = nn.Linear(self.d0, self.d)
                # Positional Encoding
                self.pe         = nn.Embedding(self.T, self.d)

        def forward(self, video_features, video_mask):
                # video_features: B x T x input_video_dim, video_mask: B x T x 1 -> x: B x T x d
                x = self.ve(video_features) * video_mask.float()

                # CHECK - WILL THIS BE A TRAINABLE PARAMETER?
                # FIX - THIS IS WRONG I GUESS, WE NEED TO USE ONE HOT ENCODING LIKE THE PAPER USED.
                p = torch.arange(0, video_mask.shape[1]).unsqueeze(0).long().to(self.device)
                p = self.pe(p) * video_mask.float()

                x += p

                return x

class QueryEncoder(nn.Module):

        def __init__(self, max_query_length = 13, lstm_hidden_size = 256):
                super(QueryEncoder, self).__init__()

                self.max_query_length = max_query_length
                self.lstm_hidden_size = lstm_hidden_size

                self.lstm = nn.LSTM(input_size = 300, hidden_size = lstm_hidden_size, num_layers = 2, bidirectional = True, batch_first = True)

        def forward(self, query_features, query_mask):
                # query_features: B x max_query_length x 300, query_mask: B x max_query_length x 1
                length = query_mask.sum(1).squeeze()

                pack_wemb = nn.utils.rnn.pack_padded_sequence(query_features, length.to('cpu'), batch_first = True, enforce_sorted = False)
                x, _ = self.lstm(pack_wemb)
                fw, max_ = nn.utils.rnn.pad_packed_sequence(x, batch_first = True, total_length = self.max_query_length)
                fw = fw.contiguous() # [B, max_query_length, 2 * lstm_hidden_size]

                B, L, H = fw.size()
                idx = (length-1).long() # 0-indexed
                idx = idx.view(B, 1, 1).expand(B, 1, H//2)
                fLSTM = fw[:,:,:H//2].gather(1, idx).view(B, H//2)
                bLSTM = fw[:,0,H//2:].view(B,H//2)
                fs = torch.cat([fLSTM, bLSTM], dim=1) # fs : [hNq->, <-h1] i.e., [B, 2 * lstm_hidden_size]

                return fs, fw

class Backbone(nn.Module):

        def __init__(self, T = 64, d = 512, input_video_dim = 1024, max_query_length = 13, lstm_hidden_size = 256, device = 'cpu'):
                super(Backbone, self).__init__()

                self.videoencoder = VideoEncoder(T, d, input_video_dim, device)
                self.queryencoder = QueryEncoder(max_query_length, lstm_hidden_size)

        def forward(self, video_features, video_mask, query_features, query_mask):
                # fv: B x T x d
                fv              = self.videoencoder(video_features, video_mask)
                # fs: B x d, fw: B x max_query_length x d
                fs, fw  = self.queryencoder(query_features, query_mask)

                # f: B x T x d
                f = fv * fs.unsqueeze(1) # Hadamard product of video and sentence features

                return f, fs, fw

# CHECK - SHOULD WE ACCOUNT FOR VIDEO MASK IN THE Wc? MOMENTS [0,15] WILL BE INVALID IF THE VIDEO IS NOT SPANNING THE WHOLE 16 OF L
# CHECK - THIS MIGHT CAUSE ISSUES WITH FM, FB, FC?
# CHECK - WHAT ABOUT THE CASE - 31 FRAMES, AND HTHE LAST VALID MOMENT IS INCLUDING A 0 FOR THE 32ND FRAME?
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
        
        # CHECK - How to deal with the case where L does not divide T?
        # CHECK - Wc is a upper triangular matrix. Should we make it symmetric as we are using softmax?
        def __init__(self, T = 64, L = 16, C = 4, device = 'cpu'):
                super(ProposalGeneration, self).__init__()

                self.T                  = T
                self.L                  = L
                self.C                  = C
                self.device             = device
                # CHECK - WILL Wc BECOME A TRAINABLE PARAMETER?
                self.Wc                 = compute_content_matrix(T, L, C).to(self.device)
                self.avg_pool   = nn.AvgPool1d(T//L, stride=T//L)

        def forward(self, f, moment_mask):
                # Wc: (L, L, C, T), f: (B, T, D), moment_mask: (B, L, L) -> fc: (B, L, L, C, D)
                fc = torch.einsum('lmit, btj -> blmij', self.Wc, f) * moment_mask.unsqueeze(-1).unsqueeze(-1)
                # fc: (B, L, L, C, D) -> fm: (B, L, L, D)
                fm = torch.mean(fc, dim=3)
                # f: (B, T, D) -> fb: (B, D, T)
                fb = torch.permute(f, (0, 2, 1))
                # fb: (B, D, T) -> fb: (B, D, L)
                fb = self.avg_pool(fb)
                # fb: (B, D, L) -> fb: (B, L, D)
                fb = torch.permute(fb, (0, 2, 1))
                return fc, fm, fb

class Attention1(nn.Module):

    def __init__(self, D):
        super(Attention, self).__init__()

        self.D, self.attn_weights = D, None
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
    
    def forward(self, query, key, value, mask = None):
        # query: (B, Lq, D) -> (B, Lq, D), key: (B, Lk, D) -> (B, Lk, D)  
        query, key = self.W_q(query), self.W_k(key)
        # attn_weights: (B, Lq, Lk)
        attn_weights = torch.matmul(query, torch.transpose(key, 2, 1))/math.sqrt(self.D)
        if mask is not None:
                # mask: (B, Lk, 1) -> (B, 1, Lk)
                mask = mask.squeeze().float().unsqueeze(1)
                # attn_weights: (B, Lq, Lk)
                attn_weights = attn_weights * mask
                # Make sure query words that are padded don't impact the attention
                attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        # attn_weights: (B, Lq, Lk)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # attn_weights: (B, Lq, Lk), value: (B, Lk=Lv, D) -> attn_out: (B, Lq, D)
        attn_out = torch.matmul(attn_weights, value)
        self.attn_weights = attn_weights
        return attn_out

class Attention(nn.Module):

    def __init__(self, D, h):
        super(Attention, self).__init__()

        self.D, self.h, self.attn_weights = D, h, None
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)

    def forward(self, query, key, value, mask = None):
        B, Lq, Lk, D = query.shape[0], query.shape[1], key.shape[1], query.shape[2]
        # query: (B, Lq, D) -> (B, Lq, D), key: (B, Lk, D) -> (B, Lk, D)
        query, key, dk = self.W_q(query), self.W_k(key), self.D // self.h
        query = query.reshape(B, Lq, self.h, dk).permute(0, 2, 1, 3)
        value = value.reshape(B, Lk, self.h, dk).permute(0, 2, 1, 3)
        key = key.reshape(B, Lk, self.h, dk).permute(0, 2, 1, 3)
        # attn_weights: (B, Lq, Lk)
        attn_weights = torch.matmul(query, torch.transpose(key, -2, -1))/math.sqrt(dk)
        if mask is not None:
                # mask: (B, Lk, 1) -> (B, 1, Lk)
                mask = mask.squeeze().float().unsqueeze(1).unsqueeze(1)
                # attn_weights: (B, Lq, Lk)
                attn_weights = attn_weights * mask
                # Make sure query words that are padded don't impact the attention
                attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        # attn_weights: (B, Lq, Lk)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # attn_weights: (B, Lq, Lk), value: (B, Lk=Lv, D) -> attn_out: (B, Lq, D))
        attn_out = torch.matmul(attn_weights, value)
        attn_out = attn_out.permute(0, 2, 1, 3)
        attn_out = attn_out.reshape(B, Lq, D)
        self.attn_weights = attn_weights
        return attn_out

class BoundaryUnit(nn.Module):

    def __init__(self, D, h):
        super(BoundaryUnit, self).__init__()

        self.D = D
        self.attn_layer = Attention(D, h)
    
    def forward(self, f_b, f_w, f_s, f_m, query_mask, length_mask):
        # length_mask: (B, L) -> f_b_mask: (B, L, 1)
        f_b_mask = length_mask.unsqueeze(-1)

        # f_b: (B, L, D), f_w: (B, Nq, D), query_mask: (B, Nq, 1) -> f_baq: (B, L, D)
        f_baq = self.attn_layer(f_b, f_w, f_w, query_mask)
        f_baq = f_baq * f_b_mask
        # f_b: (B, L, D), f_s: (B, D), f_baq: (B, L, D) -> f_bq: (B, L, D)
        f_bq = f_b*(f_baq + torch.unsqueeze(f_s, 1))
        # f_bq: (B, L, D) -> A_b: (B, L, L) 
        A_b = torch.bmm(f_bq, torch.permute(f_bq, (0, 2, 1)))/math.sqrt(self.D)

        # length_mask: (B, L) -> mask: (B, 1, L)
        mask = length_mask.float().unsqueeze(1)
                # A_b: (B, L, L), mask: (B, 1, L) -> A_b: (B, L, L)
        A_b = A_b * mask
        # Make sure moments that are padded don't impact the attention
        A_b = A_b.masked_fill(mask == 0, -1e9)
        A_b = nn.functional.softmax(A_b, dim = -1)
        # A_b: (B, L, L), f_b_mask: (B, L, 1) -> A_b: (B, L, L)
        A_b = A_b * f_b_mask

        # A_b: (B, L, L), f_b: (B, L, D) -> f_bb: (B, L, D)
        f_bb = torch.matmul(A_b, f_b)
        f_bb = f_bb * f_b_mask

        # f_m: (B, L, L, D), f_s: (B, D) -> g_m: (B, L, L, D)
        g_m = torch.sigmoid(f_m * torch.unsqueeze(torch.unsqueeze(f_s, 1), 2))
        # A_b: (B, L, L), f_m: (B, L, L, D) -> f_bm: (B, L, D)
        # I am not confident about the below line.
        f_bm = torch.sum(torch.unsqueeze(A_b, 3) * (g_m*f_m), dim = 2)

        return f_bb + f_b + f_bm

class ContentAttention(nn.Module):

    def __init__(self, D):
        super(ContentAttention, self).__init__()

        self.D, self.attn_weights = D, None
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
    
    def forward(self, query, key, value, mask = None):
        # query: (B, L, L, C, D) -> (B, L, L, C, D), key: (B, Nq, D) -> (B, 1, 1, Nq, D)
        query, key = self.W_q(query), self.W_k(key).unsqueeze(1).unsqueeze(2)
        # query: (B, L, L, C, D), key: (B, 1, 1, Nq, D) -> attn_weights: (B, L, L, C, Nq)
        attn_weights = torch.matmul(query, torch.transpose(key, 3, 4))/math.sqrt(self.D)
        if mask is not None:
                # mask: (B, Nq, 1) -> (B, 1, 1, 1, Nq)
                mask = mask.squeeze().float().unsqueeze(1).unsqueeze(2).unsqueeze(3)
                # attn_weights: (B, L, L, C, Nq)
                attn_weights = attn_weights * mask
                # Make sure query words that are padded don't impact the attention
                attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
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

        self.linear_c_hat       = nn.Linear(D, dl)
        self.linear_w_hat       = nn.Linear(D, dl)
        self.linear_s_hat       = nn.Linear(D, dl)
        self.linear_c           = nn.Linear(dl, D)
        self.attn_layer         = ContentAttention(dl)
    
    def forward(self, f_c, f_w, f_s, f_m, query_mask, moment_mask):
        # moment_mask: (B, L, L) -> f_c_mask: (B, L, L, 1, 1)
        f_c_mask = moment_mask.unsqueeze(-1).unsqueeze(-1)

        # f_c: (B, L, L, C, D), moment_mask: (B, L, L) -> f_c_hat: (B, L, L, C, dl)
        f_c_hat = self.linear_c_hat(f_c) * f_c_mask
        # f_w: (B, Nq, D), query_mask: (B, Nq, 1) -> f_w_hat: (B, Nq, dl)
        f_w_hat = self.linear_w_hat(f_w) * query_mask
        # f_s: (B, D) -> f_s_hat: (B, dl)
        f_s_hat = self.linear_s_hat(f_s)

        # f_c_hat: (B, L, L, C, dl), f_w_hat: (B, Nq, dl) -> f_caq: (B, L, L, C, dl)
        f_caq = self.attn_layer(f_c_hat, f_w_hat, f_w_hat, query_mask)
        f_caq = f_caq * f_c_mask
        # f_c: (B, L, L, C, dl), f_s_hat: (B, dl), f_caq: (B, L, L, C, dl) -> f_cq: (B, L, L, C, dl)
        f_cq = f_c_hat*(f_caq + f_s_hat.unsqueeze(1).unsqueeze(2).unsqueeze(3))
        # f_cq: (B, L, L, C, dl) -> A_c: (B, L, L, C, C)
        A_c = torch.matmul(f_cq, torch.transpose(f_cq, 3, 4))/math.sqrt(self.dl)

        # A_c: (B, L, L, C, C) -> A_c: (B, L, L, C, C)
        A_c = nn.functional.softmax(A_c, dim = -1)
        A_c = A_c * f_c_mask

        # A_c: (B, L, L, C, C), f_c_hat: (B, L, L, C, dl) -> f_cc_hat: (B, L, L, C, dl)
        f_cc_hat = torch.matmul(A_c, f_c_hat)

        # f_cc_hat: (B, L, L, C, dl) -> f_cc: (B, L, L, C, D)
        f_cc = self.linear_c(f_cc_hat) * f_c_mask

        # f_m: (B, L, L, D), f_s: (B, D) -> g_m: (B, L, L, D)
        g_m = torch.sigmoid(f_m * f_s.unsqueeze(1).unsqueeze(2))
        # g_m: (B, L, L, D), f_m: (B, L, L, D) -> fbar_m: (B, L, L, D)
        fbar_m = g_m*f_m

        return f_cc + f_c + fbar_m.unsqueeze(3)

class MomentUnit(nn.Module):

        def __init__(self, D):
                super(MomentUnit, self).__init__()

                self.D = D
                # CHECK - REVIEW THESE CONVOLUTIONAL LAYERS LATER
                self.conv_layer_fb = nn.Conv2d(D, D, 1)
                self.conv_layer_fc = nn.Conv2d(D, D, 1)

        def forward(self, f_c, f_m, f_b, moment_mask):
                # moment_mask: (B, L, L) -> f_m_mask: (B, L, L, 1)
                f_m_mask = moment_mask.unsqueeze(-1)

                # f_b: (B, L, D) -> f_b_s: (B, L, 1, D)
                f_b_s = f_b.unsqueeze(2)
                # f_b: (B, L, D) -> f_b_e: (B, 1, L, D)
                f_b_e = f_b.unsqueeze(1)
                # f_c: (B, L, L, C, D) -> f_c_mean: (B, L, L, D)
                f_c_mean = torch.mean(f_c, 3)
                # below operations don't change dimensions of the inputs
                conv_fb = self.conv_layer_fb((f_b_s * f_b_e).permute(0, 3, 1 ,2)).permute(0, 2, 3, 1)
                conv_fb = conv_fb * f_m_mask
                conv_fc = self.conv_layer_fc(f_c_mean.permute(0, 3, 1 ,2)).permute(0, 2, 3, 1)
                conv_fc = conv_fc * f_m_mask
                return conv_fb + conv_fc + f_m

class CGAEncoder1(nn.Module):
    def __init__(self, D):
        super(CGAEncoder, self).__init__()
        self.D, self.attn_weights = D, None
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
        self.W_v = nn.Linear(D, D)

        self.W_qg = nn.Linear(D, D)
        self.W_vg = nn.Linear(D, D)
        self.b_g = nn.Parameter(torch.zeros(D))
        self.b_i = nn.Parameter(torch.zeros(D))
        self.sigmoid = nn.Sigmoid()

    
    # query: (B, Nq, D), query_mask: (B, Nq), key: (B, L, L, D), value: (B, L, L, D), kv_mask: (B, L, L)
    def forward(self, query, query_mask, key, value, kv_mask):
        # key, value: (B, L, L, D) -> key, value: (B, L*L, D)
        B, L, D =  key.shape[0], key.shape[1], key.shape[3]
        key, value = torch.reshape(key, (B, L*L, D)), torch.reshape(value, (B, L*L, D))
        query, key, value  = self.W_q(query), self.W_k(key), self.W_v(value) 
        if query.dim() == 2:
            query = query.unsqueeze(1)

        # query: (B, Nq, D), key, value: (B, L*L, D) -> attn_weights: (B, Nq, L*L)
        attn_weights = torch.matmul(query, torch.transpose(key, 2, 1))/math.sqrt(D)
        # kv_mask: (B, L, L) -> kv_mask: (B, 1, L*L)
        kv_mask = kv_mask.reshape(B, L*L).float().unsqueeze(1)
        # attn_weights: (B, Nq, L*L)
        attn_weights = attn_weights * kv_mask
        # attn_weights: (B, Nq, L*L)
        attn_weights = attn_weights.masked_fill(kv_mask == 0, -1e9)
        # attn_weights: (B, Nq, L*L)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        self.attn_weights = attn_weights
        
        # attn_weights: (B, Nq, L*L), value: (B, L*L, D) -> attn_out: (B, Nq, D)
        attn_out = torch.matmul(attn_weights, value)
        if query_mask is not None:
                attn_out = attn_out * query_mask.float()

        #temp_out = self.W_qg(query) + self.W_vg(attn_out)
        #attn_out = self.sigmoid(temp_out + self.b_g) * (temp_out + self.b_i)

        if query_mask is not None:
                attn_out = attn_out * query_mask.float()
        else:
                attn_out = attn_out.squeeze(1)

        return attn_out

class CGAEncoder(nn.Module):
    def __init__(self, D, h):
        super(CGAEncoder, self).__init__()
        self.D, self.h, self.attn_weights = D, h, None
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
        self.W_v = nn.Linear(D, D)

        self.W_qg = nn.Linear(D, D)
        self.W_vg = nn.Linear(D, D)
        self.b_g = nn.Parameter(torch.zeros(D))
        self.b_i = nn.Parameter(torch.zeros(D))
        self.sigmoid = nn.Sigmoid()

    # query: (B, Nq, D), query_mask: (B, Nq), key: (B, L, L, D), value: (B, L, L, D), kv_mask: (B, L, L)
    def forward(self, query, query_mask, key, value, kv_mask):
        # key, value: (B, L, L, D) -> key, value: (B, L*L, D)
        B, L, D =  key.shape[0], key.shape[1], key.shape[3]
        key, value = torch.reshape(key, (B, L*L, D)), torch.reshape(value, (B, L*L, D))
        query, key, value  = self.W_q(query), self.W_k(key), self.W_v(value)
        if query.dim() == 2:
            query = query.unsqueeze(1)
        query_copy, Nq = query, query.shape[1]
        
        # query: (B, Nq, D), key, value: (B, L*L, D) -> attn_weights: (B, Nq, L*L)
        query = query.reshape(B, Nq, self.h, D//self.h).permute(0, 2, 1, 3)
        value = value.reshape(B, L*L, self.h, D//self.h).permute(0, 2, 1, 3)
        key = key.reshape(B, L*L, self.h, D//self.h).permute(0, 2, 1, 3)
        attn_weights = torch.matmul(query, torch.transpose(key, -2, -1))/math.sqrt(D//self.h)
        # kv_mask: (B, L, L) -> kv_mask: (B, 1, L*L)
        kv_mask = kv_mask.reshape(B, L*L).float().unsqueeze(1).unsqueeze(1)
        # attn_weights: (B, Nq, L*L)
        attn_weights = attn_weights * kv_mask
        # attn_weights: (B, Nq, L*L)
        attn_weights = attn_weights.masked_fill(kv_mask == 0, -1e9)
        # attn_weights: (B, Nq, L*L)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        self.attn_weights = attn_weights
        
        # attn_weights: (B, Nq, L*L), value: (B, L*L, D) -> attn_out: (B, Nq, D)
        attn_out = torch.matmul(attn_weights, value)
        attn_out = attn_out.permute(0, 2, 1, 3)
        attn_out = attn_out.reshape(B, Nq, D)
        if query_mask is not None:
                attn_out = attn_out * query_mask.float()
        
        #temp_out = self.W_qg(query_copy) + self.W_vg(attn_out)
        #attn_out = self.sigmoid(temp_out + self.b_g) * (temp_out + self.b_i)

        if query_mask is not None:
                attn_out = attn_out * query_mask.float()
        else:
                attn_out = attn_out.squeeze(1)

        return attn_out

class SMI(nn.Module):

        def __init__(self, D, dl, h, fian):
                super(SMI, self).__init__()

                self.D = D
                self.dl = dl
                self.fian = fian

                self.content_unit = ContentUnit(D, dl)
                self.boundary_unit = BoundaryUnit(D, h)
                self.moment_unit = MomentUnit(D)
                self.cga_encoder = CGAEncoder(D, h)

                self.content_unit1 = ContentUnit(D, dl)
                self.boundary_unit1 = BoundaryUnit(D, h)
                self.moment_unit1 = MomentUnit(D)
                self.cga_encoder1 = CGAEncoder(D, h)

        def forward(self, fc, fm, fb, fw, fs, query_mask, length_mask, moment_mask):
                # Q(1) = Encoder(Q, V)
                fw1 = self.cga_encoder(fw, query_mask, fm, fm, moment_mask)
                fs1 = self.cga_encoder(fs, None, fm, fm, moment_mask)

                # V(1) = Encoder(V, Q(1))
                fc1 = self.content_unit(fc, fw1, fs1, fm, query_mask, moment_mask)
                fb1 = self.boundary_unit(fb, fw1, fs1, fm, query_mask, length_mask)
                fm1 = self.moment_unit(fc1, fm, fb1, moment_mask)

                # V(2) = Encoder(V, Q)
                fc2 = self.content_unit1(fc, fw, fs, fm, query_mask, moment_mask)
                fb2 = self.boundary_unit1(fb, fw, fs, fm, query_mask, length_mask)
                fm2 = self.moment_unit1(fc2, fm, fb2, moment_mask)
                
                # Q(2) = Encoder(Q, V(2))
                fw2 = self.cga_encoder1(fw, query_mask, fm2, fm2, moment_mask)
                fs2 = self.cga_encoder1(fs, None, fm2, fm2, moment_mask)

                if self.fian==True:
                    fc, fm, fb = torch.max(fc1, fc2), torch.max(fm1, fm2), torch.max(fb1, fb2)
                    fw, fs = torch.max(fw1, fw2), torch.max(fs1, fs2)
                    return fc, fm, fb, fw, fs
                else:
                    return fc2, fm2, fb2, fw1, fs1

class Localization(nn.Module):

        def __init__(self, D):
                super(Localization,self).__init__()

                self.conv_layer_pm = nn.Conv2d(D, 1, 1)
                self.conv_layer_ps = nn.Conv1d(D, 1, 1)
                self.conv_layer_pe = nn.Conv1d(D, 1, 1)
                self.conv_layer_pa = nn.Conv1d(D, 1, 1)
                self.sigmoid = nn.Sigmoid()

        def forward(self, f_m, f_b, length_mask, moment_mask):
                # f_m: (B, L, L, D), moment_mask: (B, L, L) -> p_m: (B, L, L)
                p_m = self.sigmoid(self.conv_layer_pm(f_m.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)).squeeze(3) * moment_mask
                # f_b: (B, L, D), length_mask: (B, L) -> p_s: (B, L)
                p_s = self.sigmoid(self.conv_layer_ps(f_b.permute(0, 2, 1)).permute(0, 2, 1)).squeeze(2) * length_mask
                # f_b: (B, L, D), length_mask: (B, L) -> p_e: (B, L)
                p_e = self.sigmoid(self.conv_layer_pe(f_b.permute(0, 2, 1)).permute(0, 2, 1)).squeeze(2) * length_mask
                # f_b: (B, L, D), length_mask: (B, L) -> p_a: (B, L)
                p_a = self.sigmoid(self.conv_layer_pa(f_b.permute(0, 2, 1)).permute(0, 2, 1)).squeeze(2) * length_mask
                return p_m, p_s, p_e, p_a

class SMIN(nn.Module):

        def __init__(self, T, L, C, D, dl, h, fian, num_smi_layers, input_video_dim, max_query_length, lstm_hidden_size, device = 'cpu'):
                super(SMIN, self).__init__()

                self.T                                  = T
                self.L                                  = L
                self.C                                  = C
                self.D                                  = D
                self.dl                                 = dl
                self.h                                  = h
                self.fian                               = fian
                self.num_smi_layers     = num_smi_layers
                self.input_video_dim    = input_video_dim
                self.max_query_length   = max_query_length
                self.lstm_hidden_size   = lstm_hidden_size
                self.device                     = device

                self.backbone                   = Backbone(self.T, self.D, self.input_video_dim, self.max_query_length, self.lstm_hidden_size, self.device)
                self.pgm                                = ProposalGeneration(self.T, self.L, self.C, self.device)
                self.smis                               = nn.ModuleList([SMI(self.D, self.dl, self.h, self.fian) for i in range(self.num_smi_layers)])
                self.localization               = Localization(self.D)

        def forward(self, video_features, video_mask, query_features, query_mask, length_mask, moment_mask):
                f, fs, fw                               = self.backbone(video_features, video_mask, query_features, query_mask)

                fc, fm, fb                              = self.pgm(f, moment_mask)

                for i in range(self.num_smi_layers):
                        fc, fm, fb, fw, fs                      = self.smis[i](fc, fm, fb, fw, fs, query_mask, length_mask, moment_mask)

                pm, ps, pe, pa                  = self.localization(fm, fb, length_mask, moment_mask)

                return pm, ps, pe, pa
