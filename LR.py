# -*- coding: utf-8 -*-

import torchtext
from torchtext.vocab import Vectors, GloVe
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from utils import loadSST, training, testing
from tqdm import tqdm

'''
A logistic regression model
'''

### define the model ###
class LR(nn.Module):
    def __init__(self,vocab_size):
        super(LR,self).__init__()
        self.w = nn.Embedding(vocab_size,1)
        self.b = nn.Parameter(torch.zeros(1),requires_grad=True)
        
    def forward(self,batch):
        dp = torch.sum(self.w(batch.text),dim=0).squeeze()
        return dp + self.b

### load the SST dataset ###
train_iter, val_iter, test_iter, TEXT, LABEL = loadSST()

### train the model ###
model = LR(len(TEXT.vocab))
optimizer = optim.SGD(model.parameters(),lr=0.1)

num_epoch = 10
model = training(train_iter,model,num_epoch,optimizer)

### test the model ###

test_loss, accuracy_test = testing(test_iter,model)

