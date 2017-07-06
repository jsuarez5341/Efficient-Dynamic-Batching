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

def sample_gumbel(input):
    noise = torch.rand(input.size())
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return Variable(noise)

def gumbel_softmax_sample(inp, temperature):
    noise = sample_gumbel(inp)
    if inp.data.type() == 'torch.cuda.FloatTensor':
       noise = noise.cuda()
    x = inp + noise*temperature
    x = F.softmax(x)
    #x = F.log_softmax(x)
    return x.view_as(inp)

class DiffArgmax(nn.Module):
   def __init__(self, op):
      super(DiffArgmax, self).__init__()
      self.op = op

   def forward(self, x, *args):
      x = self.op(x, *args)
      x = t.max(x, len(x.size())-1)[1][:,:,0]
      return x

   def backward(self, grad_out):
      return self.op.backward(grad_out)

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

   def forward(self, x, trainable):
      q, img, ans, prog = x #Need ans for reinforce
      if not trainable: ans = None #Safety
      p = self.ProgramGenerator(q, trainable=trainable)

      batch, sLen, v = p.size() 
      #p = gumbel_softmax_sample(p, 10000)#self.temperature)
      p = p.view(-1, v)
      p = F.softmax(p)
      #p = gumbel_softmax_sample(p, temperature)#self.temperature)
      #self.temperature *= 0.9975
      p = p.view(batch, sLen, v)

      p, pInds = t.max(p, 2)

      pInds = pInds[:, :, 0]
      p= p[:, :, 0]

      a = self.ExecutionEngine((pInds, p, img))

      #p = p.view(-1)
      #pInds = pInds.view(-1)

      '''
      if trainable:
         opt.zero_grad()
         loss = criterion(a, ans)
         loss.backward(retain_variables=True)
         #cellGrads = 2*[t.sum(self.ExecutionEngine.CNN.conv1.weight.grad)] + 

         cellGrads = []
         for i in range(2):
            grad = self.ExecutionEngine.CNN.conv1.weight.grad
            if grad is None:
               cellGrads += [None]
            else:
               cellGrads += [t.sum(grad)]
         for i in range(2, pVocab):
            grad = self.ExecutionEngine.cells[i].conv1.weight.grad
            if grad is None:
               cellGrads += [None]
            else:
               cellGrads += [t.sum(grad)]

         grads = []      
         progCells = []
         for i in range(batchSz * sLen):
            cellInd = pInds.data[i]
            if cellGrads[cellInd] is not None:
               grads += [cellGrads[pInds.data[i]]]
               progCells += [p[i]]

         grads = t.stack(grads)
         progCells = t.stack(progCells)
         progCells.backward(grads.data)
         opt.step()
      '''

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
   cuda=True #All the cudas
   model = 'ExecutionEngine'
   root='saves/' + sys.argv[1] + '/'
   saver = utils.SaveManager(root)
   maxSamples = None
   
   #Hyperparams
   embedDim = 300
   eta = 1e-3

   #Params
   maxEpochs = 100
   batchSz = 640
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
      criterion = nn.CrossEntropyLoss()#utils.maskedCE
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


   #distributedBatchSz = batchSz*1
   #net = t.nn.DataParallel(net, device_ids=[1])

   if cuda:
      net.cuda()
   #net.load_state_dict(t.load(root+'weights'))
      
   opt = t.optim.Adam(net.parameters(), lr=eta)
   #opt = t.optim.Adam(filter(lambda e: e.requires_grad, net.parameters()), lr=eta)
   #opt = t.optim.Adam(net.ProgramGenerator.parameters(), lr=eta)
   
   if not validate:
      train()
   else:
      test()
'''
      #p = F.softmax(p)
      #_, p = t.max(p, 2)
      #p, pInds = p[:,:,0], pInds[:,:,0]

      #p = gumbel_softmax_sample(p, self.temperature)
      #p = t.max(p, 2)[1][:,:,0]

      #p = self.ArgMax(p, self.temperature)  
      #self.temperature *= 0.997

      
      #Breaks graph
      batch, sLen, v = p.size() 

      #flatP = p.view(-1, v)
      #flatProg = prog.view(-1)

      if trainable: #Sample
         p = F.softmax(p)
         p = p.view(-1, v)
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
         print(self.expectedReward.eda)
         reward = reward.data.expand_as(ones).contiguous().view(-1, 1)
         pReinforce.reinforce(reward - self.expectedReward.eda)

         t.autograd.backward(pReinforce, [None for _ in pReinforce])

'''
