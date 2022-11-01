import  torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18ori
import pdb

class FeatureNet(nn.Module):
    def __init__(self, args, n_class=2):
        super(FeatureNet,self).__init__()
        self.args = args
        self.n_class = n_class
        
        self.encoder = resnet18ori()
        self.pre_fc = nn.Linear(self.encoder.out_dim, self.n_class)

    def forward(self, data_shot):

        feature_map = self.encoder(data_shot).detach()
        embedding = F.adaptive_avg_pool2d(feature_map,(1,1))
        embedding = embedding.view(embedding.size(0), -1)
        logits = self.pre_fc(embedding)

        return logits, feature_map