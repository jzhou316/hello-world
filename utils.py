# -*- coding: utf-8 -*-

import torchtext
from torchtext.vocab import Vectors, GloVe
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def loadSST():
    # Our input $x$
    TEXT = torchtext.data.Field()

    # Our labels $y$
    LABEL = torchtext.data.Field(sequential=False)
    
    train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')
    
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)
    
    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10, repeat=False, device=-1)
    # should set 'repeat' to False to stop iterating forever
    
    # Build the vocabulary with word embeddings
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))
    
#    TEXT.vocab.load_vectors(vectors=GloVe(name='6B'))
    
    return train_iter, val_iter, test_iter, TEXT, LABEL

def training(train_iter,model,num_epoch,optimizer):
    loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epoch):
        total_loss = torch.Tensor([0])
        train_iter.init_epoch()
        for batch in train_iter:
            model.zero_grad()
            output = model(batch)
            loss = loss_fn(output,2 - batch.label.type(torch.FloatTensor))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data
            
            if train_iter.iterations % 100 == 0:
                print('Epoch %d/%d, iteration %d --- average loss: %f' 
                      %(epoch+1,num_epoch,train_iter.iterations,total_loss.numpy()/train_iter.iterations))
    
    print('Training completed!\n\n\n')
    
    return model

def testing(test_iter,model):
    accuracy_test = 0
    N_test = 0
    test_loss = 0
    loss_fnsum = nn.BCEWithLogitsLoss(size_average=False)
    test_iter.init_epoch()
    for batch in tqdm(test_iter):
    #    if test_iter.iterations % 100 == 0:
    #        print('Testing --- iteration %d\n' %(test_iter.iterations))
        N_test += batch.batch_size
        batch.text.volitile = True
        output = model(batch)
        test_loss += loss_fnsum(output, 2 - batch.label.type(torch.FloatTensor))
        for s in range(batch.batch_size):
            if output.data[s] >= 0  and batch.label.data[s] == 1:
                accuracy_test += 1
            elif output.data[s] < 0 and batch.label.data[s] == 2:
                accuracy_test += 1
    
    test_loss = test_loss.data[0] / N_test
    accuracy_test = accuracy_test/N_test
    print('Testing loss (average): %f \n Testing accuracy: %f' %(test_loss, accuracy_test))
    
    return test_loss, accuracy_test

    
    