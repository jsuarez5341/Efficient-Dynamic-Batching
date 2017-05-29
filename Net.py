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

from lib import utils
from ClevrBatcher import ClevrBatcher
from model.ExecutionEngine import ExecutionEngine
from model.ProgramGenerator import ProgramGenerator

#Load PTB
def dataBatcher():
   print('Loading Data...')

   trainBatcher = ClevrBatcher(batchSz, 'Train', maxSamples=500) 
   validBatcher = ClevrBatcher(batchSz, 'Val', maxSamples=100)
   print('Data Loaded.')

   return trainBatcher, validBatcher

class EndToEndBatcher():
   def __init__(self, batcher):
      self.batcher = batcher
      self.batches = batcher.batches

   def next(self):
      x, y, mask = self.batcher.next()
      q, img, imgIdx = x
      p, ans = y
      qMask, pMask = mask
      return [q, img], [ans[:, 0]], None

class ProgramBatcher():
   def __init__(self, batcher):
      self.batcher = batcher
      self.batches = batcher.batches

   def next(self):
      x, y, mask = self.batcher.next()
      q, img, imgIdx = x
      p, ans = y
      qMask, pMask = mask
      return [q], [p], pMask

class ExecutionBatcher():
   def __init__(self, batcher):
      self.batcher = batcher
      self.batches = batcher.batches

   def next(self):
      x, y, mask = self.batcher.next()
      q, img, imgIdx = x
      p, ans = y
      qMask, pMask = mask
      return [p, img], [ans[:, 0]], None


class EndToEnd(nn.Module):
   def __init__(self,
            embedDim, hGen, qLen, qVocab, pVocab,
            numUnary, numBinary, numClasses,
            dropProb):
      super(EndToEnd, self).__init__()

      self.ProgramGenerator = ProgramGenerator(
            embedDim, hGen, qLen, qVocab, pVocab)

      self.ExecutionEngine  = ExecutionEngine(
            numUnary, numBinary, numClasses)

   def forward(self, x, trainable):
      q, img = x
      p = self.ProgramGenerator(q, trainable=trainable)
      #Breaks graph
      _, p = t.max(p, 2)
      p = p[:, :, 0]
      a = self.ExecutionEngine((p, img))
      return a

def train():
   tl, ta, vl, va = [], [], [], []
   epoch = -1
   while True:
      epoch += 1
      start = time.time()

      trainLoss, trainAcc = utils.runData(net, opt, trainBatcher, 
            criterion, trainable=True, verbose=True, cuda=cuda)
      validLoss, validAcc = utils.runData(net, opt, validBatcher,
            criterion, trainable=False, verbose=False, cuda=cuda)

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

   model = 'EndToEnd'
   
   #Hyperparams
   embedDim = 300
   eta = 0.0005
   dropProb = 1.0 - 0.75

   #Params
   batchSz = 50
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
      trainBatcher = EndToEndBatcher(trainBatcher)
      validBatcher = EndToEndBatcher(validBatcher)
      criterion = nn.CrossEntropyLoss()
 
   elif model == 'ProgramGenerator':
      trainBatcher = ProgramBatcher(trainBatcher)
      validBatcher = ProgramBatcher(validBatcher)
      criterion = utils.maskedCE
      net = ProgramGenerator(
            embedDim, hGen, qLen, qVocab, pVocab)
   elif model == 'ExecutionEngine':
      trainBatcher = ExecutionBatcher(trainBatcher)
      validBatcher = ExecutionBatcher(validBatcher)
      criterion = nn.CrossEntropyLoss()
      net = ExecutionEngine(
            numUnary, numBinary, numClasses)


   #distributedBatchSz = batchSz*1
   #net = t.nn.DataParallel(net, device_ids=[1])

   if cuda:
      net.cuda()
   #net.load_state_dict(root+'weights')
      
   opt = t.optim.Adam(filter(lambda e: e.requires_grad, net.parameters()), lr=eta)
   
   train()

