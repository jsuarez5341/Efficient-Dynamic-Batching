from pdb import set_trace as T
import sys
import time
import numpy as np
import torch
import torch as t
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 

from rawLangBatcher import RawBatcher
from HyperLinear import HyperLinear

#Provides sane dataloader defaults
def ezLoader(data, batch_size, shuffle=True, num_workers=2):
   return torch.utils.data.DataLoader(data, 
         batch_size=batch_size, shuffle=shuffle, 
         num_workers=num_workers)

#Load PTB
def dataBatcher():
   print('Loading Data...')
   train = 'data/nlm/ptb.train.txt'
   valid = 'data/nlm/ptb.valid.txt'
   test  = 'data/nlm/ptb.test.txt'
   vocab = 'data/nlm/ptb.train.txt'

   trainBatcher = RawBatcher(train, vocab)
   validBatcher = RawBatcher(valid, vocab)
   testBatcher  = RawBatcher(test, vocab)
   print('Data Loaded.')

   return trainBatcher, validBatcher, testBatcher

def highwayGate(Ws, s, trainable):
   h = s.size()[1]
   hh, tt  = t.split(Ws, h, 1)
   hh, tt = F.tanh(hh), F.sigmoid(tt) 
   cc = 1 - tt
   tt = F.dropout(tt, gateDrop, trainable)
   return hh*tt + s*cc

class HyperRHN(nn.Module):

   def __init__(self):
      super(HyperRHN, self).__init__()
      #Hypernet
      self.hiddenHyper = nn.ModuleList(depth*[nn.Linear(hHyper, 2*hHyper)])
      self.inputHyper  = nn.Linear(embedDim, 2*hHyper)

      #RHN
      self.hidden = nn.ModuleList(depth*[HyperLinear(hNetwork, 2*hNetwork)])
      self.input = HyperLinear(embedDim, 2*hNetwork)

      self.upscaleProj = nn.Linear(hHyper, hNetwork)

   def forward(self, x, sNetwork, sHyper, trainable):
      for i in range(depth):
         #Hypernet
         Ws = self.hiddenHyper[i](sHyper)
         if i == 0:
            Ws += self.inputHyper(x)
         sHyper = highwayGate(Ws, sHyper, trainable)

         #Upscale
         z = self.upscaleProj(sHyper)

         #RHN
         Ws = self.hidden[i](sNetwork, z)
         if i == 0:
            Ws += self.input(x, z)
         sNetwork= highwayGate(Ws, sNetwork, trainable)

      return sHyper, sNetwork


class Network(nn.Module):

   def __init__(self):
      super(Network, self).__init__()
      self.embed   = nn.Embedding(vocab, embedDim)
      self.unembed = nn.Linear(hNetwork, vocab)
      self.HyperRHN = HyperRHN()

   def forward(self, x, trainable):
      x = self.embed(x)
      x = F.dropout(x, embedDrop, train)

      sHyper = Variable(t.zeros(batchSz, hHyper).cuda())
      sNetwork = Variable(t.zeros(batchSz, hNetwork).cuda())
      out = []

      for i in range(context):
         sHyper, sNetwork = self.HyperRHN(x[:, i], sNetwork, sHyper, trainable)
         out += [sNetwork]
         
      x = t.stack(out, 1)
      x = x.view(batchSz*context, hNetwork)
      x = self.unembed(x)
      x = x.view(batchSz, context, vocab)
      return x


def runData(batcher, trainable=False, verbose=False, minContext=0):
   m = batcher.m
   iters = int(m/distributedBatchSz/context)
   correct = 0.0
   lossAry = []
   for i in range(iters):
      if verbose and i % int(m/distributedBatchSz/context/10) == 0:
         sys.stdout.write('#')
         sys.stdout.flush()

      x, y = batcher.next(distributedBatchSz, context, minContext)
      y = y[:, minContext:].ravel()
      x, y = t.from_numpy(x).cuda(), t.from_numpy(y).cuda()
      if not trainable:
         x, y = Variable(x, volatile=True), Variable(y, volatile=True)
      else:
         x, y = Variable(x), Variable(y)

      a = net(x, trainable)

      #Mask preds
      a = a[:, minContext:].view(-1, vocab)

      loss = criterion(a, y)
      lossAry += [loss.data[0]]

      if trainable:
         opt.zero_grad()
         loss.backward()
         t.nn.utils.clip_grad_norm(net._parameters, 10.0)
         opt.step()

      #meanLossRun acc
      _, preds = t.max(a.data, 1)
      correct += sum(y.data == preds)
   acc = correct / (y.size()[0]*iters)
   meanLoss = np.mean(lossAry)

   perplexity = np.exp(meanLoss)

   return perplexity, acc

def train():
   tl, ta, vl, va = [], [], [], []
   epoch = -1
   while True:
      epoch += 1
      start = time.time()

      trainLoss, trainAcc = runData(trainBatcher, 
            trainable=True, verbose=True)
      validLoss, validAcc = runData(validBatcher)

      print('\nEpoch: ', epoch, ', Time: ', time.time()-start)
      print('| Train Perp: ', trainLoss, 
            ', Train Acc: ', trainAcc)
      print('| Valid Perp: ', validLoss, 
            ', Valid Acc: ', validAcc)

      tl += [trainLoss]
      ta += [trainAcc]
      vl += [validLoss]
      va += [validAcc]
      np.save(root+'tl.npy', tl)
      np.save(root+'ta.npy', ta)
      np.save(root+'vl.npy', vl)
      np.save(root+'va.npy', va)

      t.save(net.state_dict(), root+'weights')

if __name__ == '__main__':
   root='saves/' + sys.argv[1] + '/'
   
   #Hyperparams
   context=100
   minContext = 50
   embedDim = 27
   depth = 7
   hNetwork = 1000
   hHyper = 128
   eta = 0.001
   #Regularizers
   gateDrop = 1.0 -  0.65
   embedDrop= 1.0 - 0.75

   #Params
   batchSz = 128
   distributedBatchSz = batchSz*1
   context=100
   minContext=50
   vocab = 50

   trainBatcher, validBatcher, testBatcher = dataBatcher()

   net = Network()
   net = t.nn.DataParallel(net, device_ids=[0])
   net.cuda()
   #net.load_state_dict(root+'weights')
      
   criterion = nn.CrossEntropyLoss()
   opt = t.optim.Adam(net.parameters(), lr=eta)
   
   train()

