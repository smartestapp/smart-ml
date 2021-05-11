##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/Sha-Lab/FEAT
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Sampler for dataloader. """
import torch
import numpy as np
import pdb

class CategoriesSampler():
    """The class to generate episodic data"""
    def __init__(self, label, n_batch, n_per, n_query):
        self.n_batch = n_batch
        self.n_per = n_per
        self.n_query = n_query

        label_set = []
        for the_label in label:
            label_set.append(tuple(the_label))
        self.opt_diag = list(set(label_set))

        self.m_ind = []
        for idx,tar in enumerate(self.opt_diag):
            idxes = []
            for id_label, the_label in enumerate(label):
                if the_label == list(tar):
                    idxes.append(id_label)
            idxes = torch.from_numpy(np.array(idxes))
            self.m_ind.append(idxes)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            # classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in range(len(self.opt_diag)):
                l = self.m_ind[c]
                if len(l) > self.n_per+self.n_query:
                    pos = torch.randperm(len(l))[:self.n_per+self.n_query]
                else:
                    pos = torch.multinomial(torch.tensor([1]*len(l),dtype=torch.float), self.n_per+self.n_query, replacement=True)
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
