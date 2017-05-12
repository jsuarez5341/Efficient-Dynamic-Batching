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
from resnetTrunc import ResNet

from clevrBatcher import ClevrBatcher

#Valid padded (odd k)
def Conv2d(fIn, fOut, k):
   pad = int((k-1)/2)
   return nn.Conv2d(fIn, fOut, k, padding=pad)

#Provides sane dataloader defaults
def ezLoader(data, batch_size, shuffle=True, num_workers=2):
   return torch.utils.data.DataLoader(data, 
         batch_size=batch_size, shuffle=shuffle, 
         num_workers=num_workers)

#Load PTB
def dataBatcher():
   print('Loading Data...')

   trainBatcher = ClevrBatcher('train')
   validBatcher = ClevrBatcher('val')
   print('Data Loaded.')

   return trainBatcher, validBatcher

class UnaryModule(nn.Module):
   def __init__(self):
      super(UnaryModule, self).__init__()
      self.conv1  = Conv2d(128, 128, 3)
      self.conv2  = Conv2d(128, 128, 3)

   def forward(self, x):
      inp = x
      x = F.relu(self.conv1(x))
      x = self.conv2(x)
      x += inp
      x = F.relu(x)
      return x

class EngineClassifier(nn.Module):
   def __init__(self):
      super(EngineClassifier, self).__init__()
      self.conv1  = Conv2d(128, 512, 1)
      self.fc1    = nn.Linear(512 * 7 * 7, 1024)
      self.pool   = nn.MaxPool2d(2)
      self.fc2    = nn.Linear(1024, numClasses)

   def forward(self, x) :
      x = F.relu(self.conv1(x))
      x = self.pool(x)
      x = x.view(batchSz, -1)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x

class BinaryModule(nn.Module):
   def __init__(self):
      super(BinaryModule, self).__init__()
      self.conv1  = Conv2d(256, 128, 1)
      self.conv2  = Conv2d(128, 128, 3)
      self.conv3  = Conv2d(128, 128, 3)

   def forward(self, x1, x2):
      x = t.cat((x1, x2), 1)
      x = F.relu(self.conv1(x))
      res = x
      x = F.relu(self.conv2(x))
      x = self.conv3(x)
      x += res
      x = F.relu(x)
      return x

class CNN(nn.Module):
   def __init__(self):
      super(CNN, self).__init__()
      self.resnet = ResNet()
      for param in self.resnet.parameters():
         param.requires_grad=False

      self.conv1  = Conv2d(1024, 128, 3)
      self.conv2  = Conv2d(128, 128, 3)

   def forward(self, x):
      x = self.resnet(x)
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      return x

class ProgramGenerator(nn.Module):
   def __init__(self):
      super(ProgramGenerator, self).__init__()
      self.embed = nn.Embedding(qVocab, embedDim)
      self.LSTM  = t.nn.LSTM(embedDim, hGen, 2)
      self.proj  = nn.Linear(hGen, pVocab)

   def forward(self, x):
      x = self.embed(x)
      x, _ = self.LSTM(x)
      x = x.view(batchSz * qLen, hGen)
      x = self.proj(x)
      x = x.view(batchSz, qLen, pVocab)
      _, x = t.max(x, 2) #Program inds
      x = t.squeeze(x)
      return x


class Node():
   def __init__(self, ind, arities, cells):
      self.arity = arities[ind]
      self.ann   = cells[ind]
      self.next = [None]*self.arity

class Program():
   def __init__(self, arities, cells):
      self.arities = arities
      self.cells = cells
      self.root = None 

   #Ind is the output argmax index of the prog generator
   def addNode(self, ind, cur=None):
      if self.root is None:
         self.root = Node(ind, self.arities, self.cells)
         return

      if cur is None:
         cur = self.root

      for i in range(cur.arity):
         #Branching wrong here
         if cur.next[i] is not None: 
            if self.addNode(ind, cur.next[i]): 
               return True
         else:
            cur.next[i] = Node(ind, self.arities, self.cells)
            return True
      return False

   def build(self, inds):
      for e in inds:
         self.addNode(e)

   def execute(self, imgFeats):
      return self.executeInternal(imgFeats, cur=self.root)

   def executeInternal(self, imgFeats, cur=None):
      if cur is None or cur.arity == 0:
         return imgFeats

      inps = []
      for i in range(cur.arity):
         inps += [self.executeInternal(imgFeats, cur.next[i])]

      return cur.ann(*inps)

   def print(self):
      self.printInternal(cur=self.root)
      
   def printInternal(self, cur=None, numTabs=0):
      if cur is None or cur.arity == 0: #Scene token
         print('\t'*numTabs+'SCENE')
         return
      else: 
         print('\t'*numTabs,cur.ann)

      for i in range(cur.arity):
         self.printInternal(cur.next[i], numTabs+1)

class ExecutionEngine(nn.Module):
   def __init__(self):
      super(ExecutionEngine, self).__init__()
      self.arities = [0] + [1]*numUnary+[2]*numBinary
      unaries = [UnaryModule() for i in range(numUnary)]
      binaries = [BinaryModule() for i in range(numBinary)]
      self.cells = nn.ModuleList([None] + unaries + binaries)
      self.CNN = CNN()
      self.classifier = EngineClassifier()

   def forward(self, x, img):
      a = []
      imgFeats = self.CNN(img)

      #Can't parallelize
      for i in range(batchSz):
         prog = Program(self.arities, self.cells)
         prog.build(x[i])
         a += [prog.execute(imgFeats[i:i+1])]
      a = t.cat(a, 0)
      a = self.classifier(a)
      return a

class Network(nn.Module):
   def __init__(self):
      super(Network, self).__init__()
      self.ProgramGenerator = ProgramGenerator()
      self.ExecutionEngine  = ExecutionEngine()

   def forward(self, x, img, trainable):
      x = self.ProgramGenerator(x)
      #Breaks graph
      x = x.data
      x = self.ExecutionEngine(x, img)
      return x

def runData(batcher, trainable=False, verbose=False, minContext=0):
   m = batcher.m
   iters = int(m/distributedBatchSz)
   correct = 0.0
   lossAry = []
   for i in range(iters):
      if verbose and i % int(m/distributedBatchSz/10) == 0:
         sys.stdout.write('#')
         sys.stdout.flush()

      xi, xq, y  = batcher.next(distributedBatchSz)
      xi, xq, y  = t.from_numpy(xi).float().cuda(gpu), t.from_numpy(xq).cuda(gpu), t.from_numpy(y).cuda(gpu)
      if not trainable:
         xi, xq, y = Variable(xi, volatile=True), Variable(xq, volatile=True), Variable(y, volatile=True)
      else:
         xi, xq, y = Variable(xi), Variable(xq), Variable(y)

      a = net(xq, xi, trainable)

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
   embedDim = 300
   hGen = 256 
   eta = 0.0005
   #Regularizers
   gateDrop = 1.0 -  0.65
   embedDrop= 1.0 - 0.75

   #Params
   batchSz = 10
   distributedBatchSz = batchSz*1
   qLen = 190
   qVocab = 96
   pVocab = 70
   numUnary = 34
   numBinary = 35
   numClasses = 60
   gpu = 1

   trainBatcher, validBatcher = dataBatcher()

   net = Network()
   #net = t.nn.DataParallel(net, device_ids=[1])
   net.cuda(gpu)
   #net.load_state_dict(root+'weights')
      
   criterion = nn.CrossEntropyLoss()
   opt = t.optim.Adam(filter(lambda e: e.requires_grad, net.parameters()), lr=eta)
   
   train()

