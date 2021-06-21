import os
import time
import pprint
import torch
import numpy as np
import torch.nn.functional as F
import pdb

def set_gpu(cuda_device):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    print('Using gpu:', cuda_device)
    
def ensure_path(path):
    """The function to make log path.
    Args:
      path: the generated saving path.
    """
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


class Averager():
    """The class to calculate the average."""
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x, bs=1):
        self.v = (self.v*self.n + x*bs) / (self.n + bs)
        self.n += bs

    def item(self):
        return self.v

def count_acc(logits, label):
    """The function to calculate the .
    Args:
      logits: input logits.
      label: ground truth labels.
    Return:
      The output accuracy.
    """
    pred = F.softmax(logits, dim=1).argmax(dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return (pred == label).type(torch.FloatTensor).mean().item()

def count_mem_acc(pred_memb, label_memb):
    assert pred_memb.size() == label_memb.size()
    num_zone = pred_memb.size()[-1]
    mem = (pred_memb == label_memb).type(torch.FloatTensor).mean(dim=1)
    mem[mem<1] = 0
    return mem.mean().item()

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)