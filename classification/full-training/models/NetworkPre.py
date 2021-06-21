import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
from scipy.special import gamma

import numpy as np
import math
import pdb
from .ResNetFeat import create_feature_extractor


class FeatureNet(nn.Module):
    def __init__(self, args, mode='self', n_class=2, flag_meta=False):
        super(FeatureNet,self).__init__()
        self.args = args
        self.mode = mode
        self.n_class = n_class
        
        
        self.encoder = create_feature_extractor(args.model_type, flag_meta)
        self.pre_fc = nn.Linear(self.encoder.out_dim, self.n_class)
        self.contrast = nn.Linear(self.encoder.out_dim, self.encoder.out_dim)
        nn.init.kaiming_normal_(self.pre_fc.weight)
        nn.init.constant_(self.pre_fc.bias, 0.0)
        
        self.decoder = Decoder(args, [512, 256, 128, 64, 32]) if self.mode == 'self' else None

    def forward(self, inp):
        if self.mode in ['pre','self']:
            return self.pretrain_forward(inp)
        elif self.mode == 'adapt':
            data_shot = inp
            return self.adapt_forward(data_shot)
        else:
            raise ValueError('Please set the correct mode.')            

    def pretrain_forward(self, inp):
        if self.mode == 'pre':
            x = F.adaptive_avg_pool2d(self.encoder(inp),(1,1))
            x = x.view(x.size(0), -1)
            return self.pre_fc(x)
        elif self.mode == 'self':
            x = self.encoder(inp)
            recon = self.decoder(x)
            x = F.adaptive_avg_pool2d(x,(1,1)).view(x.size(0), -1)
            return recon,self.pre_fc(x)

    def adapt_forward(self, data_shot):
        embedding_shot = self.encoder(data_shot)
        embedding_shot = F.adaptive_avg_pool2d(embedding_shot,(1,1))
        embedding_shot = embedding_shot.view(embedding_shot.size(0), -1)
        logits = self.pre_fc(embedding_shot)
        if self.args.conlayer == 1:
            embedding_shot = self.contrast(embedding_shot)
        return logits,embedding_shot

class Decoder(nn.Module):
    def __init__(self, args, ds_feat):
        super(Decoder,self).__init__()
        decoder = []
        for idx,the_dim in enumerate(ds_feat[:-1]):
            decoder.append(nn.ConvTranspose2d(the_dim, ds_feat[idx+1], kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder.append(nn.ReLU(True))
        decoder.append(nn.ConvTranspose2d(ds_feat[-1],1,kernel_size=7, stride=2, padding=3, output_padding=1))
        decoder.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder)
        
    def forward(self,feature):
        x = self.decoder(feature)
        return x