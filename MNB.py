# -*- coding: utf-8 -*-

import torchtext
from torchtext.vocab import Vectors, GloVe
import torch
from torch.autograd import Variable
from utils import loadSST
from tqdm import tqdm

'''
Multinomial Naive Bayes
from http://www.aclweb.org/anthology/P/P12/P12-2.pdf#page=118
'''

### load the SST dataset ###
train_iter, val_iter, test_iter, TEXT, LABEL = loadSST()

### train the model ###
# initialization
vocab_size = len(TEXT.vocab)
alpha = 1
Np = 0
Nn = 0
p = torch.ones(vocab_size) * alpha
q = torch.ones(vocab_size) * alpha

# calculation of parameters
train_iter.init_epoch()
for batch in tqdm(train_iter):
#    if train_iter.iterations % 100 == 0:
#        print('Training --- iteration %d\n' %(train_iter.iterations))
    batch_size = batch.batch_size
    Np += torch.sum(batch.label.data==1)
    Nn += torch.sum(batch.label.data==2)
    for s in range(batch_size):
        ind = torch.LongTensor(list(set(batch.text.data[:,s])))
        if batch.label.data[s] == 1:
            # positive sentence
            p[ind] += 1
        elif batch.label.data[s] == 2:
            # negative sentence
            q[ind] += 1
            
b = torch.log(torch.Tensor([Np/Nn]))
w = torch.log( (p/torch.sum(p))/(q/torch.sum(q)) )
print('Training completed!\n\n\n')

### test the model ###
accuracy_test = 0
N_test = 0
test_iter.init_epoch()
for batch in tqdm(test_iter):
#    if test_iter.iterations % 100 == 0:
#        print('Testing --- iteration %d\n' %(test_iter.iterations))
    N_test += batch.batch_size
    for s in range(batch.batch_size):
        ind = torch.LongTensor(list(set(batch.text.data[:,s])))
        classifier = torch.sign(torch.sum(w[ind]) + b)
        if classifier.numpy() >= 0 and batch.label.data[s] == 1:
            accuracy_test += 1
        elif classifier.numpy() < 0 and batch.label.data[s] == 2:
            accuracy_test += 1
            
accuracy_test = accuracy_test/N_test
print('Testing accuracy: %f' %(accuracy_test))



