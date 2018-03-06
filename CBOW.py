# -*- coding: utf-8 -*-

import torchtext
from torchtext.vocab import Vectors, GloVe
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import loadSST, training, testing
from tqdm import tqdm

'''
CBOW from https://arxiv.org/pdf/1301.3781.pdf
'''

### define the model ###
class CBOW(nn.Module):
    def __init__(self,vocab_size,vec_size,vec_pre=None):
        super(CBOW,self).__init__()
        self.w = nn.Embedding(vocab_size,vec_size)
        if vec_pre is not None:
            self.w.weight.data = vec_pre
        self.linear = nn.Linear(vec_size,1)
        
    def forward(self,batch):
        embeds = torch.sum(self.w(batch.text),dim=0)
        out = F.relu(embeds)
        out = self.linear(out).squeeze()
        return out

### load the SST dataset ###
train_iter, val_iter, test_iter, TEXT, LABEL = loadSST()

### train the model ###
model = CBOW(len(TEXT.vocab),TEXT.vocab.vectors.size(1),TEXT.vocab.vectors)
optimizer = optim.SGD(model.parameters(),lr=0.1)

num_epoch = 10
model = training(train_iter,model,num_epoch,optimizer)

### test the model ###

test_loss, accuracy_test = testing(test_iter,model)


