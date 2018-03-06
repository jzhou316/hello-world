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
CNN models from http://aclweb.org/anthology/D/D14/D14-1181.pdf
'''

### define the model ###
class CNN(nn.Module):
    def __init__(self,vocab_size,vec_size,vec_pre=None,static=False):
        super(CNN,self).__init__()
        self.w = nn.Embedding(vocab_size,vec_size)
        if vec_pre is not None:
            self.w.weight.data = vec_pre
        if static:
            self.w.weight.requires_grad = False
        
        self.conv1 = nn.Conv1d(300,100,3,padding=1)
        self.conv2 = nn.Conv1d(300,100,4,padding=2)
        self.conv3 = nn.Conv1d(300,100,5,padding=2)
        
        self.drop = nn.Dropout(p=0.5)
        self.linear = nn.Linear(300,1)
        
    def forward(self,batch):
        embeds = self.w(batch.text.t())
        embeds = embeds.transpose(1,2)
        
        c1 = F.relu(self.conv1(embeds))            # size: batch_size * 100 * length of conv
        c2 = F.relu(self.conv2(embeds))
        c3 = F.relu(self.conv3(embeds))
        
        p1 = F.max_pool1d(c1,c1.size(2))           # size: batch_size * 100 * 1
        p2 = F.max_pool1d(c2,c2.size(2))
        p3 = F.max_pool1d(c3,c3.size(2))
        
        p = torch.cat((p1,p2,p3),dim=1).squeeze()  # size: batch_size * 300
        
        p = self.drop(p)
        out = self.linear(p).squeeze()
        return out
    
class CNN_2channel(nn.Module):
    def __init__(self,vocab_size,vec_size,vec_pre=None):
        super(CNN_2channel,self).__init__()
        self.w = nn.Embedding(vocab_size,vec_size)
        self.w_static = nn.Embedding(vocab_size,vec_size)
        if vec_pre is not None:
            self.w.weight.data = vec_pre
            self.w_static.weight.data = vec_pre
        self.w_static.weight.requires_grad = False
            
        self.conv1 = nn.Conv1d(300,100,3,padding=1)
        self.conv2 = nn.Conv1d(300,100,4,padding=2)
        self.conv3 = nn.Conv1d(300,100,5,padding=2)
        
        self.conv4 = nn.Conv1d(300,100,3,padding=1)
        self.conv5 = nn.Conv1d(300,100,4,padding=2)
        self.conv6 = nn.Conv1d(300,100,5,padding=2)
        
        self.drop = nn.Dropout(p=0.5)
        self.linear = nn.Linear(600,1)
        
    def forward(self,batch):
        embeds1 = self.w(batch.text.t())
        embeds1 = embeds1.transpose(1,2)
        
        embeds2 = self.w_static(batch.text.t())
        embeds2 = embeds2.transpose(1,2)
        
        c1 = F.relu(self.conv1(embeds1))            # size: batch_size * 100 * length of conv
        c2 = F.relu(self.conv2(embeds1))
        c3 = F.relu(self.conv3(embeds1))
        
        p1 = F.max_pool1d(c1,c1.size(2))           # size: batch_size * 100 * 1
        p2 = F.max_pool1d(c2,c2.size(2))
        p3 = F.max_pool1d(c3,c3.size(2))
        
        c4 = F.relu(self.conv4(embeds2))            # size: batch_size * 100 * length of conv
        c5 = F.relu(self.conv5(embeds2))
        c6 = F.relu(self.conv6(embeds2))
        
        p4 = F.max_pool1d(c4,c4.size(2))           # size: batch_size * 100 * 1
        p5 = F.max_pool1d(c5,c5.size(2))
        p6 = F.max_pool1d(c6,c6.size(2))
        
        p = torch.cat((p1,p2,p3,p4,p5,p6),dim=1).squeeze()  # size: batch_size * 600
        p = self.drop(p)
        out = self.linear(p).squeeze()
        
        return out
        

### load the SST dataset ###
train_iter, val_iter, test_iter, TEXT, LABEL = loadSST()

### train the model ###

# CNN-non-static
model = CNN(len(TEXT.vocab),TEXT.vocab.vectors.size(1),TEXT.vocab.vectors)
optimizer = optim.Adadelta(model.parameters(),lr=0.1)

# CNN-static
# model = CNN(len(TEXT.vocab),TEXT.vocab.vectors.size(1),TEXT.vocab.vectors,static=True)
# optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),lr=0.1)

# CNN-multichannel
# model = CNN_2channel(len(TEXT.vocab),TEXT.vocab.vectors.size(1),TEXT.vocab.vectors)
# optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),lr=0.1)

num_epoch = 10
model = training(train_iter,model,num_epoch,optimizer)


### test the model ###

test_loss, accuracy_test = testing(test_iter,model)
