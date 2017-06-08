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
      pMask = mask
      return [p, img], [ans[:, 0]], None


class EndToEnd(nn.Module):
   def __init__(self,
            embedDim, hGen, qLen, qVocab, pVocab,
            numUnary, numBinary, numClasses):
      super(EndToEnd, self).__init__()

      self.ProgramGenerator = ProgramGenerator(
            embedDim, hGen, qLen, qVocab, pVocab)

      self.ExecutionEngine  = ExecutionEngine(
            numUnary, numBinary, numClasses)

      #For REINFORCE
      self.expectedReward = utils.EDA()

   def forward(self, x, trainable):
      q, img, ans, prog = x #Need ans for reinforce
      if not trainable: ans = None #Safety
      p = self.ProgramGenerator(q, trainable=trainable)

      #Breaks graph
      batch, sLen, c = p.size() 

      if trainable: #Sample
         p = F.softmax(p)
         p = p.view(-1, c)
         pReinforce = p.multinomial()
         p = pReinforce.view(batch, sLen)
      else: #Argmax
         _, p = t.max(p, 2)
         p = p[:, :, 0]

      a = self.ExecutionEngine((p, img))

      #Reinforce update
      if trainable:
         ones = t.ones(batch, sLen)
         reward = (t.max(a, 1)[1] == ans).float()
         self.expectedReward.update(t.mean(reward).data[0])
         reward = reward.data.expand_as(ones).contiguous().view(-1, 1)
         pReinforce.reinforce(reward - self.expectedReward.eda)

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
   validate = False
   cuda=True #All the cudas
   model = 'ExecutionEngine'
   root='saves/' + sys.argv[1] + '/'
   saver = utils.SaveManager(root)
   maxSamples = 640 
   
   #Hyperparams
   embedDim = 300
   #eta = 5e-4
   eta = 1e-4

   #Params
   maxEpochs = 10
   batchSz = 640
   hGen = 256 
   qLen = 45
   qVocab = 96
   pVocab = 41
   numUnary = 30
   numBinary = 9
   numClasses = 29

   trainBatcher, validBatcher = dataBatcher()

   if model == 'EndToEnd':
      net = EndToEnd(
            embedDim, hGen, qLen, qVocab, pVocab,
            numUnary, numBinary, numClasses)
      trainBatcher = EndToEndBatcher(trainBatcher)
      validBatcher = EndToEndBatcher(validBatcher)
      criterion = nn.CrossEntropyLoss()
      if validate:  #hardcoded saves
         progSave = utils.SaveManager('saves/dirk/')
         execSave = utils.SaveManager('saves/nodachi/')
         progSave.load(net.ProgramGenerator)
         execSave.load(net.ExecutionEngine)
 
   elif model == 'ProgramGenerator':
      trainBatcher = ProgramBatcher(trainBatcher)
      validBatcher = ProgramBatcher(validBatcher)
      criterion = nn.CrossEntropyLoss()#utils.maskedCE
      net = ProgramGenerator(
            embedDim, hGen, qLen, qVocab, pVocab)
      if validate: saver.load(net)

   elif model == 'ExecutionEngine':
      trainBatcher = ExecutionBatcher(trainBatcher)
      validBatcher = ExecutionBatcher(validBatcher)
      criterion = nn.CrossEntropyLoss()
      net = ExecutionEngine(
            numUnary, numBinary, numClasses)
      if validate: saver.load(net)


   #distributedBatchSz = batchSz*1
   #net = t.nn.DataParallel(net, device_ids=[1])

   if cuda:
      net.cuda()
   #net.load_state_dict(root+'weights')
      
   opt = t.optim.Adam(filter(lambda e: e.requires_grad, net.parameters()), lr=eta)
   #opt = t.optim.Adam(net.ProgramGenerator.parameters(), lr=eta)
   
   if not validate:
      train()
   else:
      test()

