##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Model for meta-transfer learning. """
import  torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_mtl import ResNetMtl
import numpy as np
import pdb


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.middle = 1000
        self.vars = nn.ParameterList()

        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
        
        # self.fc1_w = nn.Parameter(torch.ones([self.middle, self.z_dim]))
        # torch.nn.init.kaiming_normal_(self.fc1_w)
        # self.fc1_b = nn.Parameter(torch.zeros(self.middle))
        # self.fc2_w = nn.Parameter(torch.ones([self.args.way, self.middle]))
        # torch.nn.init.kaiming_normal_(self.fc2_w)
        # self.fc2_b = nn.Parameter(torch.zeros(self.args.way))

    def forward(self, input_x, the_vars=None):
        # if the_vars is None:
        #     the_vars = self.vars
        # fc1_w = the_vars[0]
        # fc1_b = the_vars[1]
        net = F.linear(input_x, self.fc1_w, self.fc1_b)
        # net = F.linear(net, self.fc2_w, self.fc2_b)
        return net

    def parameters(self):
        self.vars = nn.ParameterList()
        self.vars.append(self.fc1_w)
        self.vars.append(self.fc1_b)
        return self.vars

class Decoder(nn.Module):
    def __init__(self, args, ds_feat):
        super(Decoder,self).__init__()
        decoder = []
        for idx,the_dim in enumerate(ds_feat[:-1]):
            decoder.append(nn.ConvTranspose2d(the_dim, ds_feat[idx+1], kernel_size=3, stride=2, padding=1, output_padding=1 if idx>1 else (1,0))) #)
            decoder.append(nn.ReLU(True))
        decoder.append(nn.ConvTranspose2d(ds_feat[-1],1,kernel_size=(5,7), stride=2, padding=(2,3), output_padding=1))
        decoder.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder)
        
    def forward(self,feature):
        x = self.decoder(feature)
        return x

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='meta', num_cls=2):
        super().__init__()
        self.args = args
        self.mode = mode
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        z_dim = 640
        self.base_learner = BaseLearner(args, z_dim)

        if self.mode == 'meta':
            self.encoder = ResNetMtl()
        elif self.mode == 'cloud':
            self.encoder = ResNetMtl(mtl=False)
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif self.mode == 'cloud_ss':
            self.encoder = ResNetMtl()
            # self.pre_fc = nn.Sequential(nn.Linear(640, 1000), nn.ReLU(), nn.Linear(1000, num_cls))
            self.decoder = Decoder(args, [640, 320, 160, 80])
        else:
            self.encoder = ResNetMtl(mtl=False)  
            self.pre_fc = nn.Sequential(nn.Linear(640, 1000), nn.ReLU(), nn.Linear(1000, num_cls))
            self.decoder = Decoder(args, [640, 320, 160, 80])

    def forward(self, inp):
        """The function to forward the model.
        Args:
          inp: input images.
        Returns:
          the outputs of MTL model.
        """
        if self.mode in ['pre','self'] :
            return self.pretrain_forward(inp)
        elif self.mode=='meta':
            data_shot, label_shot, data_query = inp
            return self.meta_forward(data_shot, label_shot, data_query)
        elif self.mode=='preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        elif self.mode in ['cloud','cloud_ss']:
            # data_shot, _, _ = inp
            data_shot = inp
            return self.cloud_forward(data_shot)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        """The function to forward pretrain phase.
        Args:
          inp: input images.
        Returns:
          the outputs of pretrain model.
        """
        if self.mode == 'pre':
            x = F.adaptive_avg_pool2d(self.encoder(inp),(1,1))
            x = x.view(x.size(0), -1)
            return self.pre_fc(x)
        elif self.mode == 'self':
            return self.decoder(self.encoder(inp))


    def meta_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-train phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        embedding_query = F.adaptive_avg_pool2d(embedding_query,(1,1))
        embedding_query = embedding_query.view(embedding_query.size(0), -1)
        embedding_shot = self.encoder(data_shot)
        embedding_shot = F.adaptive_avg_pool2d(embedding_shot,(1,1))
        embedding_shot = embedding_shot.view(embedding_shot.size(0), -1)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, self.update_step):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)        
        return logits_q

    def preval_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-validation during pretrain phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        embedding_query = F.adaptive_avg_pool2d(embedding_query,(1,1))
        embedding_query = embedding_query.view(embedding_query.size(0), -1)
        embedding_shot = self.encoder(data_shot)
        embedding_shot = F.adaptive_avg_pool2d(embedding_shot,(1,1))
        embedding_shot = embedding_shot.view(embedding_shot.size(0), -1)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, 100):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)         
        return logits_q

    def cloud_forward(self, data_shot):
        embedding_shot = self.encoder(data_shot)
        embedding_shot = F.adaptive_avg_pool2d(embedding_shot,(1,1))
        embedding_shot = embedding_shot.view(embedding_shot.size(0), -1)
        logits = self.base_learner(embedding_shot) # pre_fc
        return logits