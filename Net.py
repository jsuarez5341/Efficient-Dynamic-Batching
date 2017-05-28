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
from torch.autograd import StochasticFunction

import utils
from clevrBatcher import ClevrBatcher
from ExecutionEngine import ExecutionEngine
from ProgramGenerator import ProgramGenerator

#Load PTB
def dataBatcher():
   print('Loading Data...')

   trainBatcher = ClevrBatcher(batchSz, 'train')
   validBatcher = ClevrBatcher(batchSz, 'val')
   print('Data Loaded.')

   return trainBatcher, validBatcher

class ProgramBatcher():
   def __init__(self, batcher):
      self.batcher = batcher
      self.batches = batcher.batches

   def next(self):
      x, y = self.batcher.next()
      q, img = x
      p, ans = y
      return [q], [p]

class EndToEnd(nn.Module):
   def __init__(self,
            embedDim, hGen, qLen, qVocab, pVocab,
            numUnary, numBinary, numClasses,
            dropProb):
      super(Network, self).__init__()

      self.ProgramGenerator = ProgramGenerator(
            embedDim, hGen, qLen, qVocab, pVocab)

      self.ExecutionEngine  = ExecutionEngine(
            numUnary, numBinary, numClasses)

   def forward(self, x, trainable):
      q, img = x
      q = self.ProgramGenerator(q, trainable=trainable)
      #Breaks graph
      q = q.data
      a = self.ExecutionEngine(q, img)
      return a

def train():
   tl, ta, vl, va = [], [], [], []
   epoch = -1
   while True:
      epoch += 1
      start = time.time()

      trainLoss, trainAcc = utils.runData(net, opt, trainBatcher, 
            trainable=True, verbose=True, cuda=cuda)
      validLoss, validAcc = utils.runData(net, opt, validBatcher,
            trainable=False, verbose=False, cuda=cuda)

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
   cuda=True #All the cudas

   model = 'ProgramGenerator'
   
   #Hyperparams
   embedDim = 300
   eta = 0.0005
   dropProb = 1.0 - 0.75

   #Params
   batchSz = 2
   hGen = 256 
   qLen = 45
   qVocab = 96
   pVocab = 41
   numUnary = 34
   numBinary = 35
   numClasses = 60

   trainBatcher, validBatcher = dataBatcher()

   if model == 'EndToEnd':
      net = EndToEnd(
            embedDim, hGen, qLen, qVocab, pVocab,
            numUnary, numBinary, numClasses,
            dropProb)
   elif model == 'ProgramGenerator':
      trainBatcher = ProgramBatcher(trainBatcher)
      validBatcher = ProgramBatcher(validBatcher)
      net = ProgramGenerator(
            embedDim, hGen, qLen, qVocab, pVocab)
   elif model == 'ExecutionEngine':
      net = ExecutionEngine(
            numUnary, numBinary, numClasses)


   #distributedBatchSz = batchSz*1
   #net = t.nn.DataParallel(net, device_ids=[1])

   if cuda:
      net.cuda()
   #net.load_state_dict(root+'weights')
      
   weight = t.ones(pVocab)
   weight[0] = 0
   criterion = nn.CrossEntropyLoss(weight=weight)
   opt = t.optim.Adam(filter(lambda e: e.requires_grad, net.parameters()), lr=eta)
   
   train()
