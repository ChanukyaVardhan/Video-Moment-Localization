#from dataset import *
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CharadesSTA
from models import *

#dataset
data = CharadesSTA()
train_dataloader = DataLoader(data, batch_size = 64, shuffle = True, collate_fn = data.collate_fn)
batch = next(iter(train_dataloader))
video_features, query_features, video_mask = batch['video_features'], batch['query_features'], batch['video_mask']
start_pos, end_pos = batch['start_pos'], batch['end_pos']

#model - forward pass
#Backbone - convert these to assert later
Backbone = Backbone()
backbone_out = Backbone(video_features,video_mask, query_features)
print('printing backbone output shapes f, fs, fw')
print(backbone_out[0].shape) #[64,64,512] == f video embedding -- intersting that all have 512 only. reminds so much of clip
print(backbone_out[1].shape) #[64,512] == fs sentence
print(backbone_out[2].shape) #[64,13,512] == fw word wise
f, fs, fw = backbone_out
#Proposal Generation 
ProposalGeneration = ProposalGeneration()
proposal_out = ProposalGeneration(f)
fc, fm, fb = proposal_out
print('printing proposal output shapes fc, fm, fb')
print(fc.shape) #([64, 16, 16, 4, 512]) #N, L, L, C, D
print(fm.shape) #([64, 16, 16, 512]) #N, L, L, D
print(fb.shape) #([64, 16, 512]) #N, L, D
#From paper Implementation details section
#others like hidden size of lstm, setence truncation to 13 are taken care by Chan already
D = 512
dl = 128
C = 4
L = 16
#Content Unit
ContentUnit = ContentUnit(D, dl)
cu = ContentUnit(fc, fw, fs, fm)
print('content unit output shape')
print(cu.shape) #[64, 16, 16, 4, 512] #N, L, L, C, D == fc shape
#Boundary Unit
BoundaryUnit = BoundaryUnit(D)
bu = BoundaryUnit(fb, fw, fs, fm)
print('boundary unit output shape')
print(bu.shape) #[64, 16, 512] #N, L, D == fb shape
#Moment Unit
myMomentUnit = MomentUnit(L)
mu = myMomentUnit(fc, fm, fb)
print('moment unit output shape')
print(mu.shape)
#SMI
SMI = SMI(D, dl, L)
mu3, bu3 = SMI(fc, fm, fb, fw, fs)
print('smi layer output shape mu3 and bu3')
print(mu3.shape)
print(bu3.shape)
#Localization
Loc = Localization(D)
pm, ps, pe = Loc(mu3, bu3)
print('Localization layer output shape pm, ps, pe')
print(pm.shape)
print(ps.shape)
print(pe.shape)
print('sample values of pm, ps and pe - all values should be around 0.5 since the network is untrained')
print(pm[0])
print(ps[0])
print(pe[0]) 

#Loss Function Test

