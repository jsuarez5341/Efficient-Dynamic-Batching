from pdb import set_trace as T 
import sys
import time
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
from torch.autograd import StochasticFunction

from lib import utils
from ClevrBatcher import ClevrBatcher
from model.ExecutionEngine import ExecutionEngine
from model.ProgramGenerator import ProgramGenerator

#Load PTB
def dataBatcher(maxSamples):
   print('Loading Data...')

   trainBatcher = ClevrBatcher(batchSz, 'Train', maxSamples=maxSamples) 
   validBatcher = ClevrBatcher(batchSz, 'Val', maxSamples=maxSamples)
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
      pMask = mask[0]
      return [q, img, ans[:, 0], p], [ans[:, 0]], None

class ProgramBatcher():
   def __init__(self, batcher):
      self.batcher = batcher
      self.batches = batcher.batches

   def next(self):
      x, y, mask = self.batcher.next()
      q, img, imgIdx = x
      p, ans = y
      pMask = mask[0]
      return [q], [p], pMask

class ExecutionBatcher():
   def __init__(self, batcher):
      self.batcher = batcher
      self.batches = batcher.batches

   def next(self):
      x, y, mask = self.batcher.next()
      q, img, imgIdx = x
      p, ans = y
      muls = (p*0+1.0).astype(np.float32)
      pMask = mask
      return [p, muls, img], [ans[:, 0]], None

class EndToEnd(nn.Module):
   def __init__(self,
            embedDim, hGen, qLen, qVocab, pVocab,
            numUnary, numBinary, numClasses):
      super(EndToEnd, self).__init__()

      self.ProgramGenerator = ProgramGenerator(
            embedDim, hGen, qLen, qVocab, pVocab)

      self.ExecutionEngine  = ExecutionEngine(
            numUnary, numBinary, numClasses)
   
      self.ArgMax = DiffArgmax(gumbel_softmax_sample)
      self.temperature = 1.0

      #For REINFORCE
      self.expectedReward = utils.EDA()

   def forward(self, x, trainable, fast=True):
      q, img, ans, prog = x #Need ans for reinforce
      if not trainable: ans = None #Safety

      p = self.ProgramGenerator(q)

      #Finicky handling of PG-EE transition
      batch, sLen, v = p.size() 
      p = p.view(-1, v)
      p = F.softmax(p)
      p = p.view(batch, sLen, v)
      p, pInds = t.max(p, 2)
      pInds = pInds[:, :, 0]
      p= p[:, :, 0]

      a = self.ExecutionEngine((pInds, p, img), fast=fast)
      return a

def train():
   epoch = -1
   while epoch < maxEpochs:
      epoch += 1

      start = time.time()
      trainLoss, trainAcc = utils.runData(net, opt, trainBatcher, 
            criterion, trainable=True, verbose=True, cuda=cuda)
      validLoss, validAcc = utils.runData(net, opt, validBatcher,
            criterion, trainable=False, verbose=False, cuda=cuda)
      trainEpoch = time.time() - start

      print('\nEpoch: ', epoch, ', Time: ', trainEpoch)
      print('| Train Perp: ', trainLoss, 
            ', Train Acc: ', trainAcc)
      print('| Valid Perp: ', validLoss, 
            ', Valid Acc: ', validAcc)

      saver.update(net, trainLoss, trainAcc, validLoss, validAcc)

def test():
   start = time.time()
   validLoss, validAcc = utils.runData(net, opt, validBatcher,
         criterion, trainable=False, verbose=True, cuda=cuda)

   print('| Valid Perp: ', validLoss, 
         ', Valid Acc: ', validAcc)
   print('Time: ', time.time() - start)


if __name__ == '__main__':
   load = False
   validate = False
   cuda = True #All the cudas
   fast = True #Parallel
   model = 'ExecutionEngine'
   root='saves/' + sys.argv[1] + '/'
   saver = utils.SaveManager(root)
   maxSamples = None
   
   #Hyperparams
   embedDim = 300
   eta = 1e-4

   #Params
   maxEpochs = 200
   batchSz = 64
   hGen = 256 
   qLen = 45
   qVocab = 96
   pVocab = 41
   numUnary = 30
   numBinary = 9
   numClasses = 29

   trainBatcher, validBatcher = dataBatcher(maxSamples)

   if model == 'EndToEnd':
      net = EndToEnd(
            embedDim, hGen, qLen, qVocab, pVocab,
            numUnary, numBinary, numClasses)
      trainBatcher = EndToEndBatcher(trainBatcher)
      validBatcher = EndToEndBatcher(validBatcher)
      criterion = nn.CrossEntropyLoss()
      if load:  #hardcoded saves
         progSave = utils.SaveManager('saves/cpyprog9k/')
         execSave = utils.SaveManager('saves/cpyEE/')
         progSave.load(net.ProgramGenerator)
         execSave.load(net.ExecutionEngine)
 
   elif model == 'ProgramGenerator':
      trainBatcher = ProgramBatcher(trainBatcher)
      validBatcher = ProgramBatcher(validBatcher)
      criterion = nn.CrossEntropyLoss()
      net = ProgramGenerator(
            embedDim, hGen, qLen, qVocab, pVocab)
      if load: saver.load(net)

   elif model == 'ExecutionEngine':
      trainBatcher = ExecutionBatcher(trainBatcher)
      validBatcher = ExecutionBatcher(validBatcher)
      criterion = nn.CrossEntropyLoss()
      net = ExecutionEngine(
            numUnary, numBinary, numClasses)
      if load: saver.load(net)


   if cuda:
      net.cuda()
      
   opt = t.optim.Adam(net.parameters(), lr=eta)
   #opt = t.optim.Adam(filter(lambda e: e.requires_grad, net.parameters()), lr=eta)
   
   if not validate:
      train()
   else:
      test()
